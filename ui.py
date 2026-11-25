#!/usr/bin/env python3
"""Simple local web UI for Book Writing Agent OS (standard library only).

    python ui.py            # then open http://127.0.0.1:8765  (opens automatically)

It runs the same Pipeline as main.py in a background thread. Whenever the pipeline needs the
author (approve the plan, approve a chapter), the page shows the text with Approve / Revise / Stop.
Listens on 127.0.0.1 only, so nobody else on the network can reach it.
"""
from __future__ import annotations

import argparse
import os
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bookos.env import load_env

ENV_FILE = load_env()

from bookos.llm import get_llm
from bookos.orchestrator import BOOK_TYPES, Pipeline, validate_setup

OUT_DIR = Path("book_data")


class Run:
    """One pipeline run plus the state the page needs. All access goes through `lock`."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.status = "idle"  # idle | running | waiting | done | error | stopped
        self.log: list[str] = []
        self.gate: dict | None = None
        self.decision: tuple[str, str] | None = None
        self.event = threading.Event()
        self.error = ""
        self.root: Path | None = None
        self.usage: dict = {}
        self.thread: threading.Thread | None = None

    # ---- callbacks used by the pipeline thread
    def add_log(self, msg: str) -> None:
        with self.lock:
            self.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def approver(self, kind: str, content: str) -> tuple[str, str]:
        with self.lock:
            self.gate = {"kind": kind, "content": content}
            self.decision = None
            self.status = "waiting"
            self.event.clear()
        self.event.wait()
        with self.lock:
            decision = self.decision or ("quit", "")
            self.gate = None
            self.status = "running"
        return decision

    # ---- controls used by HTTP handlers
    def start(self, setup: dict, chapters: int, mock: bool, provider: str) -> None:
        with self.lock:
            if self.status in ("running", "waiting"):
                raise RuntimeError("A run is already in progress.")
            self.reset()
            self.status = "running"
        self.thread = threading.Thread(target=self._work, args=(setup, chapters, mock, provider), daemon=True)
        self.thread.start()

    def _work(self, setup: dict, chapters: int, mock: bool, provider: str) -> None:
        try:
            setup = validate_setup(setup)
            llm = get_llm(mock, provider or None)
            pipe = Pipeline(llm, OUT_DIR, self.approver, log=self.add_log)
            mem = pipe.run(setup, chapters_to_write=chapters)
            with self.lock:
                self.root = mem.root
                self.usage = llm.usage_report()
                self.status = "done"
        except SystemExit as e:  # plan not approved, or missing API key
            with self.lock:
                self.error = str(e.code or "Stopped.")
                self.status = "stopped"
        except Exception as e:  # noqa: BLE001 - show any failure to the user; progress is saved on disk
            with self.lock:
                self.error = f"{type(e).__name__}: {e}"
                self.status = "error"

    def decide(self, decision: str, feedback: str) -> None:
        with self.lock:
            if self.status != "waiting":
                raise RuntimeError("Nothing is waiting for a decision.")
            self.decision = (decision, feedback)
        self.event.set()

    def snapshot(self) -> dict:
        with self.lock:
            files = []
            if self.root and self.root.exists():
                files = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file() and "drafts" not in p.parts)
            return {
                "status": self.status, "log": self.log[-200:], "gate": self.gate, "error": self.error,
                "root": str(self.root) if self.root else "", "usage": self.usage, "files": files,
            }

    def manuscript(self) -> str:
        with self.lock:
            p = (self.root / "manuscript.md") if self.root else None
        return p.read_text(encoding="utf-8") if p and p.exists() else ""


RUN = Run()


def has_key(provider: str) -> bool:
    var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    return bool(os.getenv(var))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            snap = RUN.snapshot()
            snap["key_ok"] = {"openai": has_key("openai"), "anthropic": has_key("anthropic")}
            snap["env_file"] = str(ENV_FILE) if ENV_FILE else ""
            self._json(snap)
        elif self.path == "/api/example":
            p = Path("examples/setup_nonfiction.json")
            self._json(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})
        elif self.path == "/manuscript.md":
            text = RUN.manuscript()
            self._send(200 if text else 404, text.encode("utf-8"), "text/markdown; charset=utf-8",
                       {"Content-Disposition": 'attachment; filename="manuscript.md"'})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        try:
            data = self._body()
            if self.path == "/api/start":
                setup = {k: str(data.get(k, "")).strip() for k in ("book_type", "idea", "target_reader", "length", "style", "author_notes")}
                if not setup["author_notes"]:
                    setup.pop("author_notes")
                validate_setup(setup)  # fail fast with a clear message instead of after starting
                RUN.start(setup, max(1, int(data.get("chapters") or 2)), bool(data.get("mock")), str(data.get("provider") or ""))
            elif self.path == "/api/decide":
                RUN.decide(str(data.get("decision")), str(data.get("feedback", "")))
            else:
                return self._send(404, b"not found", "text/plain")
            self._json({"ok": True})
        except (RuntimeError, ValueError, json.JSONDecodeError) as e:
            self._json({"ok": False, "error": str(e)}, 400)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Book Writing Agent OS</title>
<style>
:root{--ink:#1c1c28;--mute:#666;--line:#dcdce6;--brand:#3730a3;--bg:#f6f6fb;--ok:#0a7a4b;--bad:#b42318}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{background:var(--brand);color:#fff;padding:14px 24px}header h1{margin:0;font-size:19px}header p{margin:2px 0 0;opacity:.85;font-size:13px}
main{max-width:920px;margin:20px auto;padding:0 16px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:16px}
h2{margin:0 0 12px;font-size:16px}label{display:block;font-weight:600;font-size:13px;margin:12px 0 4px}
input,select,textarea{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:6px;font:inherit;background:#fff}
textarea{min-height:70px;resize:vertical}.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1;min-width:180px}
button{font:inherit;font-weight:600;border:0;border-radius:6px;padding:9px 16px;cursor:pointer;background:var(--brand);color:#fff}
button.sec{background:#fff;color:var(--ink);border:1px solid var(--line)}button.ok{background:var(--ok)}button.danger{background:#fff;color:var(--bad);border:1px solid var(--bad)}
button:disabled{opacity:.5;cursor:not-allowed}.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;align-items:center}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:700;background:#eee}
.pill.running{background:#e0e7ff;color:var(--brand)}.pill.waiting{background:#fef3c7;color:#92400e}.pill.done{background:#d1fae5;color:var(--ok)}.pill.error,.pill.stopped{background:#fee2e2;color:var(--bad)}
pre{margin:0;white-space:pre-wrap;word-wrap:break-word;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace}
.doc{background:#fafafd;border:1px solid var(--line);border-radius:6px;padding:14px;max-height:480px;overflow:auto}
.log{background:#14141f;color:#d7d7e8;border-radius:6px;padding:12px;max-height:240px;overflow:auto}
.warn{background:#fef3c7;color:#92400e;border-radius:6px;padding:8px 12px;font-size:13px;margin-top:10px}.hint{color:var(--mute);font-size:12px;margin-top:4px}
.err{background:#fee2e2;color:var(--bad);border-radius:6px;padding:8px 12px;margin-bottom:10px}.hide{display:none}
</style></head><body>
<header><h1>Book Writing Agent OS</h1><p>Plan, write and check a book with a team of AI agents. You approve the plan and every chapter.</p></header>
<main>
<div id="setup" class="card">
  <h2>1. Book setup</h2>
  <div class="row"><div><label>Book type</label><select id="book_type"></select></div>
  <div><label>Chapters to write now</label><input id="chapters" type="number" min="1" value="2"></div></div>
  <label>Idea</label><textarea id="idea" placeholder="What is the book about?"></textarea>
  <div class="row"><div><label>Target reader</label><input id="target_reader"></div>
  <div><label>Length</label><input id="length" value="40,000 words, 12 chapters"></div></div>
  <label>Style / voice</label><input id="style" placeholder="e.g. Warm, practical, short paragraphs">
  <label>Facts, people or must-include material (optional)</label><textarea id="author_notes"></textarea>
  <div class="hint">For memoir especially: the writer only uses facts you give here. Gaps are marked [AUTHOR TO CONFIRM].</div>
  <div class="row"><div><label>Provider</label><select id="provider"><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option></select></div>
  <div><label>&nbsp;</label><label style="font-weight:400;margin:0"><input type="checkbox" id="mock" style="width:auto"> Demo mode (offline, no API key, placeholder text)</label></div></div>
  <div id="keywarn" class="warn hide"></div>
  <div class="actions"><button id="start">Start</button><button class="sec" id="example">Fill with example</button>
  <span class="hint">Runs resume automatically: same idea = same folder, already approved work is reused.</span></div>
</div>
<div id="err" class="err hide"></div>
<div id="progress" class="card hide"><h2>2. Progress <span id="pill" class="pill"></span></h2><div class="log"><pre id="log"></pre></div></div>
<div id="gatecard" class="card hide"><h2 id="gatetitle"></h2>
  <div class="doc"><pre id="gatebody"></pre></div>
  <label>Feedback (needed only for Revise)</label><textarea id="feedback" placeholder="e.g. Make chapter 2 more concrete and cut the second example."></textarea>
  <div class="actions"><button class="ok" id="approve">Approve</button><button id="revise">Revise with feedback</button><button class="danger" id="stop">Stop (progress is saved)</button></div>
  <div class="hint">To change the text yourself, edit the draft file named at the top of a chapter review, save it, then click Approve. Your edits are kept.</div>
</div>
<div id="donecard" class="card hide"><h2>3. Result</h2><div id="doneinfo"></div>
  <div class="actions"><a href="/manuscript.md"><button>Download manuscript.md</button></a><button class="sec" id="toggle">Show / hide manuscript</button></div>
  <div id="ms" class="doc hide" style="margin-top:12px"><pre id="mstext"></pre></div>
</div>
</main>
<script>
const $=id=>document.getElementById(id);
const TYPES=%TYPES%;
TYPES.forEach(t=>{const o=document.createElement('option');o.value=o.textContent=t;$('book_type').appendChild(o)});
$('book_type').value='non-fiction';
let lastGate=null,msLoaded=false,keyOk={},envFile='';
async function post(p,b){const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});const j=await r.json();if(!j.ok)throw new Error(j.error||'failed');return j}
function showErr(m){$('err').textContent=m;$('err').classList.toggle('hide',!m)}
function keyWarn(){const w=$('keywarn'),p=$('provider').value;
  if(!$('mock').checked&&keyOk[p]===false){w.textContent='No '+(p==='openai'?'OPENAI_API_KEY':'ANTHROPIC_API_KEY')+' found. '+(envFile?'Read '+envFile+' but the key is missing or empty there.':'No .env file found next to ui.py.')+' Add the key, restart, or tick Demo mode.';w.classList.remove('hide')}else w.classList.add('hide')}
$('provider').onchange=$('mock').onchange=keyWarn;
$('example').onclick=async()=>{const e=await (await fetch('/api/example')).json();for(const k in e){if($(k))$(k).value=e[k]}};
$('start').onclick=async()=>{showErr('');msLoaded=false;
  try{await post('/api/start',{book_type:$('book_type').value,idea:$('idea').value,target_reader:$('target_reader').value,length:$('length').value,
    style:$('style').value,author_notes:$('author_notes').value,chapters:$('chapters').value,mock:$('mock').checked,provider:$('provider').value})}catch(e){showErr(e.message)}};
const decide=(d)=>async()=>{const fb=$('feedback').value.trim();if(d==='revise'&&!fb){showErr('Type what to change first.');return}
  showErr('');try{await post('/api/decide',{decision:d,feedback:fb});$('feedback').value=''}catch(e){showErr(e.message)}};
$('approve').onclick=decide('approve');$('revise').onclick=decide('revise');
$('stop').onclick=()=>{if(confirm('Stop now? Approved chapters and drafts are saved; start again with the same idea to resume.'))decide('quit')()};
$('toggle').onclick=async()=>{const m=$('ms');m.classList.toggle('hide');if(!msLoaded){$('mstext').textContent=await (await fetch('/manuscript.md')).text();msLoaded=true}};
async function tick(){
  try{const s=await (await fetch('/api/state')).json();keyOk=s.key_ok;envFile=s.env_file;keyWarn();
    const active=s.status==='running'||s.status==='waiting';
    $('setup').classList.toggle('hide',active);$('start').disabled=active;
    $('progress').classList.toggle('hide',s.status==='idle');$('pill').textContent=s.status;$('pill').className='pill '+s.status;
    const lg=$('log');const atEnd=lg.parentElement.scrollTop+lg.parentElement.clientHeight>=lg.parentElement.scrollHeight-30;
    lg.textContent=s.log.join('\\n');if(atEnd)lg.parentElement.scrollTop=lg.parentElement.scrollHeight;
    $('gatecard').classList.toggle('hide',!s.gate);
    if(s.gate){const key=s.gate.kind+s.gate.content.length;if(key!==lastGate){lastGate=key;
      $('gatetitle').textContent=s.gate.kind==='plan'?'Approve the chapter plan':'Review '+s.gate.kind;$('gatebody').textContent=s.gate.content;
      $('gatecard').scrollIntoView({behavior:'smooth'})}}else lastGate=null;
    if(s.status==='error'||s.status==='stopped')showErr((s.status==='error'?'The run failed: ':'')+s.error+(s.status==='error'?'  Progress is saved; press Start again with the same setup to resume.':''));
    $('donecard').classList.toggle('hide',s.status!=='done');
    if(s.status==='done'){const u=s.usage||{};$('doneinfo').innerHTML='Saved in <code>'+s.root+'</code><br>Files: '+s.files.join(', ')+
      '<br>Model calls: '+(u.calls||0)+' · tokens in/out: '+(u.input_tokens||0)+' / '+(u.output_tokens||0)+(u.estimated_cost_usd!==undefined?' · est. cost $'+u.estimated_cost_usd:'')}
    if(s.status==='done'||s.status==='error'||s.status==='stopped')$('setup').classList.remove('hide');
  }catch(e){}
}
setInterval(tick,1000);tick();
</script></body></html>""".replace("%TYPES%", json.dumps(list(BOOK_TYPES)))


def main() -> None:
    p = argparse.ArgumentParser(description="Local web UI for Book Writing Agent OS")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"Book Writing Agent OS UI running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
