# Pre-registration: cross-session site re-identification (the polypectomy-site question)

Committed BEFORE any of the ten runs is tracked. Clinical motivation: at surveillance
colonoscopy, the endoscopist must re-find the site of a previous polypectomy. Phantom
form of the question: if a wall site is marked in procedure A's monocular map, does an
INDEPENDENTLY tracked procedure B over the same segment place that site at the same
location?

## Data (fixed now, from data properties only)

Qualified pairs: same segment, two independent captures (v1 vs v4 texture/lighting
version, same trajectory style; GT paths 27-39 mm apart, so genuinely different camera
paths through the same anatomy). Session A = v1 capture, session B = v4 capture.

1. c1_ascending_t4_v1_s1  vs c1_ascending_t4_v4
2. c1_cecum_t1_v1         vs c1_cecum_t1_v4
3. c1_descending_t4_v1    vs c1_descending_t4_v4
4. c2_cecum_t1_v1         vs c2_cecum_t1_v4
5. c2_transverse1_t1_v1_s1 vs c2_transverse1_t1_v4_s1

DISQUALIFIED on a measured data property (before any tracking): the two mixed-trajectory
pairs (c1_sigmoid1 t1-vs-t4, c1_transverse1 t1-vs-t4), whose GT trajectories sit ~320 mm
apart - their ground-truth frames do not overlap, so no cross-session comparison is
defined for them.

## Protocol (frozen)

- Each of the 10 runs is tracked independently by the standard adopted pipeline
  (c3vdv2_screening_seq.py: undistortion, fine-tuned depth, v2 robust tracking,
  divergence guard), blind to its partner.
- ORACLE REGISTRATION, stated up front exactly as in the loop-closure oracle: each run's
  estimated trajectory is Sim(3)-aligned (Umeyama) to its own ground truth, placing both
  sessions in the shared segment frame. This experiment therefore measures the CEILING of
  cross-session map agreement; automatic in-the-wild session registration is explicitly
  out of scope and named as future work.
- Sites (9 per pair, 45 total), deterministic: in session A, marking frames at 30%, 50%,
  70% of the undistorted frame count (floor); at each, pixels at image center and at
  horizontal offsets of +/- one quarter width; any pixel without valid estimated depth
  shifts outward to the nearest valid pixel. Site position = back-projection through
  session A's aligned estimated pose and estimated depth.
- Re-identification: the site is projected into every session-B aligned estimated camera;
  among frames where it lands in the central 80% of the image with valid depth, the
  closest camera is selected; session B's OWN estimated depth and aligned pose at that
  pixel back-project to the re-found position. A site visible in no session-B frame
  counts as not re-found (infinite error).
- Error per site = Euclidean distance (mm) between the session-A and session-B positions.

## Gates

- **G1 (accuracy):** median error over all 45 sites <= 10 mm - the scale of a
  polypectomy scar region, i.e. close enough to direct re-inspection to the right patch.
- **G2 (discrimination):** for >= 80% of sites, the re-found position is nearer to its
  own session-A site than to any other of that pair's sites.

## Pre-stated outcomes

- Both pass: claim cross-session site localization at the measured median error under
  oracle registration, with session registration the named remaining step.
- Any gate fails: the result is published as the oracle bound on cross-session map
  agreement with a mechanism analysis - a negative here is a finding about monocular map
  stability, exactly as the loop-closure oracle was.
- No site rule, pair list, threshold, or gate changes after this commit.

---

## VERDICT (results section, added after the experiment ran - Oct 17 2026)

**Both gates FAIL; the pre-stated negative outcome applies.** Across 45 sites in 5 pairs:
median cross-session error **26.3 mm** vs the 10 mm bar (G1 FAIL); nearest-own
discrimination **29%** vs 80% (G2 FAIL); 10/45 sites never re-found in session B's
usable view. Per-pair medians 16.1 / 15.9 / 35.4 / 21.3 / 28.5 mm - a uniform
negative, not one bad pair.

**Mechanism:** cross-session site error is floored by per-session reconstruction error
at the site. Each session contributes screening-scale trajectory drift (per-run Sim(3)
alignment residuals 0.9-16 mm RMSE) plus monocular depth error at the marked pixel,
compounded across two independent sessions and widened by the v1-to-v4 appearance shift
the depth network must absorb. This is the project's characterized residual-drift
limitation measured in its clinical costume: with today's per-session maps, a mark made
in one procedure lands 2-4 cm away in the next procedure's map, an order above the
polypectomy-scar scale, even when session registration is handed to the system for free.

**Published claim:** the oracle bound on monocular cross-session site re-identification
for this pipeline is ~26 mm median; the binding constraint is per-session map accuracy,
not session registration. The route forward is named, not implied: either substantially
better per-session maps or appearance-based site re-recognition, both beyond this
project's scope. The negative mirrors the loop-closure oracle in method and in value.
