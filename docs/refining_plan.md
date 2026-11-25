# Part 2: Plan for refining an existing book

**Principle:** the system proposes small, reviewable edits and the author decides. It never rewrites a chapter wholesale and never adds facts.

**1. Input.** Accept .docx, .md, .txt. Convert to one internal format: Markdown with stable IDs. .docx via python-docx (headings, paragraphs, italics kept); .md/.txt parsed directly. Split by headings into chapters, then chapters into paragraphs, each with an ID like `ch07-p23` and a content hash. If there are no headings, detect chapters by patterns ("Chapter N", page breaks) or ask the author to confirm the split. The original file is never modified.

**2. Finding problems.** Two passes.
*Book-level pass (once):* for every chapter an LLM extracts a short summary, names, terms, facts and a style profile (sentence length, person, tense, tone). This builds the same memory the writing system uses: a "book bible".
*Chapter-level pass:* each chapter is checked against the bible. Rule-based checks find name/spelling variants, repeated sentences and stories (n-gram or embedding similarity), and style outliers. An LLM flags weak flow, unclear passages and contradictions with stored facts, each with the exact quote and a reason. Findings are ranked by severity.

**3. Suggesting changes.** Each finding becomes a *suggestion*: paragraph ID, before text, after text, type, reason. Rules to protect the author: edit the smallest span possible; the prompt includes style samples from the author's own chapters; no new facts, numbers or quotes (if a fix needs one, output a question to the author instead); a validator rejects suggestions that add proper nouns or numbers absent from the original.

**4. Approval.** Suggestions appear as tracked-change style diffs, grouped by chapter and type. The author accepts, rejects or edits each. Decisions are stored (accepted/rejected, with the rejected pattern remembered so it is not proposed again). Accepted edits are applied to a copy, exported as .docx with tracked changes or clean .md.

**5. Ripple effect.** Each accepted fix can create a *rule* ("Dr. Patel -> Dr. Patil", "remove the repeated coffee anecdote"). The system searches all other chapters, exact match first, then fuzzy/embedding match for paraphrased versions, and queues matches as new suggestions for the same approve/reject flow. Changing a fact updates the bible, and chapters that cite that fact are rechecked. Content hashes mean only changed paragraphs and their dependants are re-analysed.

**6. Full-length books (300+ pages).** Never send the whole book. Use hierarchical processing: paragraph -> chapter summary -> book bible (a few thousand tokens). Each chapter call gets that chapter's text, the bible, and the summaries of its neighbours; long chapters are processed in overlapping chunks. Repetition across distant chapters uses embeddings (vector search), not context. Work runs in the background in parallel by chapter with progress saved, so a failure resumes instead of restarting. Cost stays roughly linear in book length, with prompt caching on the bible.
