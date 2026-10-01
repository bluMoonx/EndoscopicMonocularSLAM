# Distillation protocol v2 — stated before any external evaluation has run

Date: 2026-09-30. At this moment, no model in this project has ever been
evaluated on CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, or CVC-300.
This file is committed before the first such evaluation.

## Why the v1 accuracy gate is being revised (and the others are not)

The v1 gate G1 (student val Dice >= 0.95) asked the student to match the
teacher's 0.959 on our held-out 200 Kvasir-SEG images. That bar inherits a
caveat this project has documented since the batch study: the teacher is a
fine-tune of a borrowed polyp segmenter whose author's training split is
undocumented — its base may have trained on some or all of our "held-out"
val images. The student, initialized from ImageNet, has no such advantage.
Published honest-split DeepLabV3+ results on Kvasir-SEG cluster near
0.88–0.93, where the student converged in two independent runs (0.880, 0.888).

So v1's accuracy gate may compare a possibly-contaminated number against an
honest one. The fair comparison is head-to-head on data NEITHER model has
seen: the four standard external polyp benchmarks above. Precedent for a
documented re-freeze: the Trap-4 operating-point revision (conf 0.9 -> 0.7,
justified on controls, then frozen again).

This is a revision, not an erasure: the v1 result (student fails v1-G1)
remains in results/distill_student.json and in every document. If the
external head-to-head shows the same ~0.07 gap as the val split, the
contamination hypothesis is FALSIFIED, the student loses honestly, and it
stays rejected under v2 as well.

## Gates v2 (all must pass; decided on the run-4 candidate, defined below)

- **G1' external accuracy:** pooled mean Dice gap (teacher − student) <= 0.03
  across the four external sets combined, AND no single set with gap > 0.05.
  Both models zero-shot, same preprocessing each was trained with, Dice on
  binarized masks (threshold 0.5) at each dataset's ground-truth resolution.
- **G2 (unchanged):** held-out control fire rate <= 7.5% on the same two
  held-out calibration control clips (teacher: 5.0%).
- **G3 (unchanged):** >= 2.5x teacher latency, quiet machine, one ONNX
  session alive at a time, median of 50 (the only protocol that survived the
  timing-traps audit).

## The run-4 candidate (design fixed now, on speed grounds alone)

DeepLabV3+/EfficientNet-B0 at 320x240 input (measured 8.0 ms vs teacher
26.2 ms = 3.28x, PASS with margin; the 384x288 student measured 2.47x and
is excluded by G3). Training recipe = run 3's (seed 7, frozen 800/200 split,
480 hard negatives, strong augmentation, teacher soft labels, lr 1e-4
cosine, 30 epochs) with the student input at 320x240 and teacher
probabilities bilinearly resized to the student grid; unlabeled C3VD distill
pool raised to 2000 frames.

## Pilot (hypothesis test, not a gate decision)

The run-3 student (384x288) will be evaluated on the external sets alongside
the teacher, labeled as a pilot: it tests the contamination hypothesis and
burns nothing, because the run-4 design above is already fixed and its gate
decision uses only run-4's numbers.

## Adoption path (unchanged from v1)

Passing v2 does not ship the student. Adoption additionally requires
re-deriving the frozen operating point on the calibration split with the
student, and a single read of the untouched test split. Only if the test
read is within the teacher's documented performance band does the student
replace the teacher in Argus Lite.
