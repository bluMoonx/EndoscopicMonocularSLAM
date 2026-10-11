# Pre-registration: dense screening protocol (detection campaign, fresh-clip test3)

Committed BEFORE any test3 clip is downloaded. Follows results/huber_port_gate.md's
discipline: criterion first, stop rules stated, no tuning after this commit.

## Candidate (frozen as of this commit)

Dense-400 screening: sample up to 400 frames per clip (vs the published protocol's 80),
every detection gate byte-identical to batch_run.py (fire bar 0.5% of frame, conf >= 0.7,
size 2-22 mm, static-overlay zeroing), support = maxC (largest camera-frame cluster,
10 mm radius), operating point **maxC_dense >= 14** — derived once on the original calib
split under the original zero-false-alarm rule (calib control max = 6, margin 8) and not
touched again. Detector weights unchanged (polyp_ft/dlv3p_ft_ep8). Development evidence
(the burned 41 clips, results/detect_dense.csv): 16/21 polyps vs 11/21, control flags
8/20 of which 6 carry real pathology per the dataset's own annotations (cancer, UC,
duodenal ulcer w/ visible vessel, EMR site, anastomotic-leak cavity, dye injection);
clean false alarms 2 ("nothing", "blind end"). Rejected sibling: time-binned support
(14/21, margin only 2). Prior rejected lever (train_detector2/polyp_ft2): augmentation
fine-tune traded control fire rate for dyed-polyp dice; stays rejected.

## Fresh test (test3)

- 25 polyp + 25 control clips, selected by the SAME deterministic rule as test2
  (batch_fetch2.py): the next-smallest unseen annotated clips under 45 MB, split label
  "test3". No clip influences any choice; the operating point is frozen above.
- BOTH protocols run on test3: the published 80-frame protocol (threshold 14) and
  dense-400 (threshold 14). Paired comparison on identical clips.
- Controls are classified clean vs pathology-bearing by the dataset's own annotation
  text, mechanically: an annotation of "nothing" or naming only anatomy (e.g. "blind
  end", "ileocecal valve", "normal") = clean; any named finding/procedure = pathology.
  The classification is derived from the annotation file only, before results exist.

## Gates (adopt dense-400 as the published screening protocol iff ALL pass)

- **G1 (recall):** dense-400 detects strictly more test3 polyp clips than the 80-frame
  protocol does on the same clips.
- **G2 (clean false alarms):** dense-400 flags at most 2 more CLEAN controls than the
  80-frame protocol on the same clips, and at most 4 clean controls in total.
- **G3 (significance):** Fisher two-sided p < 0.05 for dense-400's polyp-vs-control
  2x2 on test3 (all controls counted as controls, pathology-bearing included —
  the conservative reading).

## Pre-stated outcomes

- All pass -> dense-400 becomes the published screening protocol; the 80-frame 11/21
  result stays on record beside it as the first frozen test; both test3 tallies
  published, flagged controls' annotations disclosed verbatim.
- Any gate fails -> the 80-frame protocol stays published as-is; the campaign is
  recorded with the dev numbers and the failed gate, mechanism investigated.
- No threshold, gate, or selection rule may be revised after this commit; a second
  candidate would require a fresh pre-registration and a FRESH test4.
