# The late project, organized as five questions

*A study aid for me, not a submission document. The manuscript and field guide tell the story in
chronological order, which is why the back half feels like a pile of disconnected experiments.
It isn't. After the four-stage pipeline existed (depth → tracking → fusion → coverage map),
nothing new was ever built. Every experiment from the mid-project reboot to the final commit
answers one of exactly five questions. When a judge asks about anything from the late project,
first decide which question it belongs to, then answer inside that frame.*

---

## Q1. How accurate is it, and how do I know WHICH stage causes the error?

The tool: the **staged ablation**. On the phantom (the only data with ground truth), swap each
estimated input for its ground-truth version, one at a time. If the error stays, the swapped
stage was innocent; if the error collapses, that stage was the bottleneck. No guessing.

- The ablation convicted **learned depth** as the sole bottleneck. Domain fine-tuning closed it:
  depth accuracy 68% → 92–99.7%, false blind-spot alarms 7–41% of wall → 0.5–3.9%.
- Final accuracy: **0.42 mm** mean tracking error, **1.3–2.4 mm** surface error,
  **98.6%** blind-spot agreement against the perfect-tracking twin.
- Every trajectory number is printed beside a trivial "constant point" baseline so a flattering
  but meaningless number cannot hide.

## Q2. Is my from-scratch tracker worth anything next to the standard tool?

Two pre-registered head-to-heads against ORB-SLAM3, rules committed to git before either ran:

- **Phantom:** a draw. 0.42 vs 0.39 mm, per-sequence 2–2–1, confidence intervals overlap.
- **Real tissue (EndoSLAM ex-vivo porcine):** a four-way statistical tie (39.6–41.4 mm across
  every tracker and variant). The error floor on real tissue comes from the data, not the tracker.

So why does my tracker exist? **Density.** ORB-SLAM3 poses only sparse keyframes and loses
5–35% of some sequences entirely; mine poses **every frame with zero tracking loss**, and a
coverage map is impossible without a pose for every frame. The honest claim is never "more
accurate than the standard tool." It is: equally accurate, while doing the thing the coverage
map actually requires.

## Q3. What breaks it, and are the breaks fixable or fundamental?

Each limitation was attacked until it produced a measured verdict. None of them is vague
"future work" — each one ends in a mechanism:

- **Rushing:** tracking collapses at ~2 effective fps, holds at ~9, fully solved at native
  rate. Real scopes stream 25+, so this is a solved requirement, not a flaw.
- **Whole-colon drift (~100 mm):** loop closure cannot fix it, proven with an oracle — even
  PERFECT loop closures recover at most 14%, because a withdrawal path is topologically a
  line, not a loop. There is nothing to close.
- **Tissue deformation:** costs ~9× accuracy. Six candidate fixes, each with a pass bar
  written down first; all six failed their bars (best miss: −18% against a 20% bar). Rejected
  with mechanisms, not abandoned.
- **Marking a site for the NEXT procedure:** lands ~26 mm away vs the ~10 mm clinical need.
  The binding constraint is per-session map error, not registration. Closed as out of reach.

## Q4. Does it survive contact with real clinical video?

- **First frozen test** (41 unseen clips): 11/21 polyp clips flagged, 3/20 "controls" flagged —
  and every control flag was real pathology (p = 0.020). The miss autopsy traced 8 of 10
  misses to sparse frame sampling, not to blind detection.
- That autopsy produced a candidate fix (dense sampling), which was **pre-registered and
  tested on 48 fresh clips**: 19/25 vs 13/25 on identical clips, p = 0.0012 → adopted. The
  flagged "controls" there were a labeled cancer, a tumor, a leak cavity, and an EMR site.
- The pattern appeared twice — generic AI failed in-body in the depth network AND in the
  detector, and targeted adaptation rescued both. **"Competence ends at the edge of the
  training data" is the project's central finding**, demonstrated twice in two different
  components.

## Q5. Why should anyone trust these numbers?

- Every experiment's pass/fail criteria were **committed to public git before the run**,
  testing stops at the first failed gate, and every outcome publishes, including failures
  with their mechanisms. The record: **23 iterations, 2 adoptions, 2 self-caught retractions.**
- A **red-team audit** re-derives every headline number with independently written code
  (different alignment algorithm, fresh parsers): 15/15, tracking ATEs match to 0.0005 mm,
  and a fresh end-to-end re-track reproduces the 0.42 mm mean digit for digit.

---

## The whole late project in one sentence

> "I built the system in four stages, then spent twenty-three pre-registered iterations trying
> to break it, comparing it to the standard tool, and proving which of its limits are bugs
> (frame rate: fixed), which are laws (drift: a withdrawal is a line), and whether anyone
> should believe me (independent re-derivation says yes)."

---

## Anticipated attacks

*Each attack a sharp judge might raise, with its honest verdict: **measured** (I already ran
that experiment and published the number), **misreads the claim** (the attack argues against
something I never said), or **real gap** (concede it, in these words).*

**"Depth networks confuse brightness with distance and hallucinate on wet, textureless walls."**
Verdict: **measured — it is my central finding.** Off-the-shelf depth scored 68% and falsely
flagged up to 41% of the wall; domain fine-tuning took it to 92–99.7% and under 4%. The real
gap inside this attack: I fine-tuned on phantom data, so generalization to diverse real human
colons is unproven, and the data to prove it does not publicly exist. I say that on the
limitations slide.

**"Claiming the tracker never loses track is impossible for monocular VO in soft tissue."**
Verdict: **misreads the claim.** I never claim the pose is always accurate. I claim the tracker
outputs a pose for every frame and never resets the way a keyframe system does, and when the
pose becomes unreliable the guard mechanism flags it — the guard-rejection rate is a published
metric, and the frame-rate experiment exists precisely to quantify where brightness alignment
fails (16–23% guard coasting at ~2 effective fps, disclosed).

**"Tissue deforms; the colon is not rigid."**
Verdict: **measured.** Deformation costs ~9× accuracy; six pre-registered fixes all failed
their own bars (best miss −18% vs a 20% bar) and were published with mechanisms.

**"Drift accumulates over a full withdrawal; the map tears apart."**
Verdict: **measured.** ~100 mm whole-colon drift, and the loop-closure oracle proves even
perfect closures recover at most 14%, because a withdrawal path is a line. This is also why
site re-identification landed at 26 mm vs the 10 mm clinical need.

**"Real-time full-colon fusion on standard hardware is an open problem."**
Verdict: **measured and agreed.** The phone runs ~3 fps against a measured 9 fps requirement
for live use, which is why the shipped application is a retrospective report card, not live
guidance. Live guidance is stated future work.

**"Three video frames does not mean a patch was inspected."**
Verdict: **partially fair — exposure, not inspection.** View counting measures whether the
mucosa was exposed to the camera (in frame, unoccluded, repeatedly), which is necessary but
not sufficient for inspection. The gap between high and low detectors may come from technique
within the areas that are seen — I said exactly this in correspondence with Dr. Rex. Coverage
is a stratification signal, not a diagnosis.

**"Constant blind-spot alerts would cause alert fatigue."**
Verdict: **misreads the claim.** Argus does not fire live alerts. The product is a
retrospective per-procedure report card for quality review. And if roughly a quarter of the
wall genuinely goes under-inspected, that is the argument for measuring it, not for ignoring it.

**"Real systems use shape-sensing hardware or live polyp detection instead."**
Verdict: **different design points, not refutations.** Shape sensors are added hardware — my
design constraint is none. Live CADe answers a different question and by construction cannot
see wall that never appears on screen.

## Words to never say

The project's credibility is that every claim is exactly its measured size. These phrasings
inflate it and will be caught:

- "Never loses track" → *"poses every frame; unreliable stretches are flagged by the guard."*
- "It caught pathology the doctors missed" → *"every flagged control contained real, labeled
  pathology, so false alarms on clean tissue were effectively zero."* (The labels were known.)
- "My tracker beats ORB-SLAM3" → *"a statistical tie on accuracy; mine poses every frame,
  which the coverage map requires."* Fit, not victory.
- "11 of 21 polyps" → *"11 of 21 polyp clips."* (Clips, not individual polyps.)
- "False alarms fell from 41%" → *"from up to 41%, depending on sequence."*
- "Re-found months later" → *"in a second, independent procedure's map."*
- "Argus Studio is built for doctors" → *"Studio is my demo and evidence console; the report
  card is a research output."*
