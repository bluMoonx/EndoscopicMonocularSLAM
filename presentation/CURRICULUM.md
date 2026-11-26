# The lock-in curriculum — Nov 25, 2026 → fair day, Mar 10, 2027

Two goals, in order of importance:
1. **UNDERSTAND ARGUS** — explain all of it, closed-book, with the right numbers.
2. **UN-AI-IFY ARGUS** — every judge-visible sentence written by you: deck, notebook, app, Studio, docs.

Everything else (admin, demos, rehearsal) hangs off those two.

## Hard deadlines (from the official GSDSEF schedule — these don't move)

| Date | What |
|---|---|
| Dec 7 | Student account created on gsdsef.org — DONE (account exists from SRC submission) |
| Dec 9 / Dec 14 | SRC safety submission + approval — DONE (submitted and approved months ago) |
| **Feb 1** | **Digital Presentation due (grades 9–12 quality screening) — THE NEXT DEADLINE** |
| Feb 12 | Resubmission deadline if screening requires changes |
| Feb 19 | Application and payment |
| Feb 22 | zFairs invite arrives |
| **Feb 26** | **Digital Presentation PDF + Digital Notebook PDF uploaded to zFairs** |
| Mar 9 | Setup at Balboa Park Activity Center, 1:00–7:00 PM |
| Mar 10 | Judging, 8:00 AM–3:00 PM |

The critical path: **the deck must be fully yours by late January**, because Feb 1 is a submission, not a draft deadline. That pulls the whole rewrite forward.

## The daily spine (every day, ~15 min, non-negotiable)

- **10 min:** flashcards from [NUMBERS_CARD.md](NUMBERS_CARD.md). Until the numbers are automatic.
- **5 min:** say one JUDGE_MAP item out loud, closed-book. Cycle through all 13 forever.

This is the whole "understanding Argus" engine. The weeks below just add deliverables on top.

---

## Week 0 · Nov 25–30 — Setup

- [ ] Laptop migration, steps 0–3 of LAPTOP_SETUP.md (one evening; done forever)
- [ ] Make the flashcards (paper or Anki) from NUMBERS_CARD.md
- [x] GSDSEF account + SRC safety approval — already done months ago. December is all study and rewrite, no admin

## Week 1 · Dec 1–7 — UNDERSTAND: the five questions, part 1

- [ ] Closed-book: Q1 (accuracy + staged ablation) and Q2 (vs ORB-SLAM3) until clean

## Week 2 · Dec 8–14 — UNDERSTAND: the five questions, part 2

- [ ] Closed-book: Q3 (what breaks it), Q4 (clinical video), Q5 (why trust the numbers)
- [ ] UN-AI: open TEXT_REWRITE.md, make the six ⚠ factual decisions first (0.91-vs-0.42 is the big one)

## Week 3 · Dec 15–21 — UNDERSTAND: the eight attacks · UN-AI: the app

- [ ] Closed-book: all 8 anticipated attacks from JUDGE_MAP (verdict + answer for each)
- [ ] UN-AI: finish every YOURS: line in TEXT_REWRITE.md and hand it back → Claude patches app + Studio and rebuilds
- [ ] Milestone check: can you do all 13 items cold? If not, the deck rewrite waits — understanding feeds writing

## Week 4 · Dec 22–28 (break week — the big rewrite starts) — UN-AI: deck part 1

- [ ] Tell Claude you're starting → build_deck.py gets retired (so it can never overwrite your words)
- [ ] Closed-book rewrite: Methods slides first (template requires student-written procedures), then the abstract (≤250 words, check every number against NUMBERS_CARD)
- [ ] Rule: never edit an AI sentence in place. Read → close → write from nothing → copy numbers back → read aloud

## Week 5 · Dec 29–Jan 4 — UN-AI: deck part 2

- [ ] Closed-book rewrite: conclusion, discussion, introduction
- [ ] Read-aloud pass on every slide. Stumble = rewrite that sentence
- [ ] Captions: keep or simplify, your call (they're mostly data)

## Week 6 · Jan 5–11 — Deck QA + first mock judging

- [ ] Template compliance check: ≤12 slides, Arial only, ≥12pt everywhere, abstract ≤250, no links, no placeholder boxes
- [ ] Mock judging round 1: Claude plays the judge, fires the five questions, you answer cold
- [ ] Fix whatever the mock exposed

## Week 7 · Jan 12–18 — UN-AI: notebook

- [ ] Closed-book rewrite of the notebook's Abstract, Introduction, and Conclusion (the judge-read pages)
- [ ] Skim every other section for sentences that don't sound like you; rewrite those
- [ ] Insert all images from Appendix C's checklist into the placeholders

## Week 8 · Jan 19–25 — Finals week for both documents

- [ ] Deck: final read-through, then DONE — nothing changes after this week
- [ ] Notebook: finish images, export a draft PDF, check page flow
- [ ] Draft the 2-minute speech (now easy — the sentences already exist in your head from the rewrites)

## Week 9 · Jan 26–Feb 1 — SUBMIT THE DECK

- [ ] **Submit the Digital Presentation before Feb 1**
- [ ] Buffer for anything week 8 left loose
- [ ] Start memorizing the 2-minute speech (the seven wow facts)

## Week 10 · Feb 2–8 — PERFORM: speech + hostile mock

- [ ] 2-minute speech memorized, timed, said out loud daily
- [ ] Mock judging round 2: the 8 attacks, hostile mode
- [ ] Watch for quality-screening feedback; fixes due Feb 12

## Week 11 · Feb 9–15 — Screening fixes + demo rehearsal begins

- [ ] Apply any screening modifications and resubmit by Feb 12
- [ ] Booth demo loop: Studio launch → live pipeline → blind-spot reveal → phone app → Studies → watch-it-build → benchmark button. Ten runs. Time it
- [ ] ADMIN: application + payment ready (due Feb 19)

## Week 12 · Feb 16–22 — Paperwork week

- [ ] ADMIN: application and payment submitted by Feb 19
- [ ] zFairs invite arrives Feb 22 — complete the profile same day
- [ ] Notebook final PDF exported; deck PDF exported

## Week 13 · Feb 23–Mar 1 — UPLOADS + laptop dry run

- [ ] **Upload Digital Presentation PDF + Digital Notebook PDF to zFairs by Feb 26**
- [ ] Laptop: git pull, then the FULL dry-run checklist from LAPTOP_SETUP.md, WiFi off, every station
- [ ] Safety net: screen-record one full live-demo run; copy to laptop + USB; print QR codes

## Week 14 · Mar 2–8 — Dress rehearsal week

- [ ] Daily: 2-min speech + demo loop + one random attack question, back to back, out loud
- [ ] Prepare the physical display per fair rules; pack the kit (laptop, charger, phone, USB, printed abstract)
- [ ] Sleep. Seriously

## Mar 9 — Setup day (1:00–7:00 PM, BPAC)

- [ ] Set up, run Studio once on-site, pin localhost:7860, leave everything charging

## Mar 10 — Fair day

- [ ] Drop-off 7:15–8:00 AM. Judging 8:00 AM–3:00 PM
- [ ] WiFi off, Studio launched, phone unlocked, speech in your head, numbers card in your pocket
- [ ] You built it, you broke it on purpose, you audited it, and you can explain all of it. Go

---

*Status of the shared work: TEXT_REWRITE.md waits on your YOURS: lines (week 2–3). The notebook draft exists and waits on your closed-book pass (week 7). Claude's jobs on request: patch-back + rebuild, retire build_deck.py, mock judging rounds, screening-fix support.*
