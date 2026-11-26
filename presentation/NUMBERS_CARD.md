# The numbers card — every figure worth memorizing, grouped by the question it answers

*Flashcard source. Front = the left column phrase, back = the number. If a judge asks anything
quantitative, the answer is on this card or it isn't worth quoting. Full provenance for every
figure: README.md results table, ITERATION_HISTORY.md, presentation/JUDGE_MAP.md.*

## The problem (why anyone cares)
| | |
|---|---|
| Precancerous polyps missed in standard colonoscopy | **~1 in 4** |
| Wall genuinely under-seen at screening pace (Argus's measurement) | **~40%** of mapped wall |

## Q1 — How accurate is it? (all vs. C3VD phantom ground truth)
| | |
|---|---|
| Camera tracking error, mean (pipeline v2, 5 sequences) | **0.42 mm** |
| Same figure before the v2 upgrade — and still the PHONE's number | **0.91 mm** |
| 3D surface accuracy (median Chamfer, 4/5 sequences) | **1.3–2.4 mm** |
| Blind-spot agreement with the perfect-tracking twin | **98.6%** |
| Depth accuracy, off-the-shelf → fine-tuned (held-out) | **68% → 92–99.7%** |
| False blind-spot alarms, off-the-shelf → fine-tuned | **7–41% of wall → 0.5–3.9%** |
| Monocular sizing error (3–30 mm targets), median | **0.60 mm** |

## Q2 — Versus the standard tool (both tests pre-registered)
| | |
|---|---|
| Phantom head-to-head: Argus vs ORB-SLAM3 | **0.42 vs 0.39 mm — a draw** (per-sequence 2–2–1, CIs overlap) |
| Real tissue (EndoSLAM pig colon): all trackers, all variants | **39.6–41.4 mm — four-way tie** |
| ORB-SLAM3 sequence coverage (sparse keyframes, loses track) | **65–95%** |
| Argus sequence coverage | **every frame, zero loss** |

## Q3 — What breaks it (each limit ends in a mechanism)
| | |
|---|---|
| Whole-colon drift over a complete withdrawal | **~100 mm** |
| Drift recoverable by PERFECT loop closure (oracle) | **≤14%** — a withdrawal is a line |
| Tracking cost of deforming tissue | **~9×** |
| Deformation fixes attempted / passed (pre-registered bars) | **6 / 0** (best miss: −18% vs a +20% bar) |
| Cross-session site re-ID: achieved vs clinically needed | **~26 mm vs ~10 mm** — closed as out of reach |
| Frame rate: tracking fails / holds / solved | **~2 fps fails (16–23% guarded) · ~9 holds (1.5–4.2%) · native solves (0.7–2.1%)** — real scopes stream 25+ |
| Flagship full withdrawal: path error over path length | **92 mm over 7,003 mm (1.3%)** vs **133 mm** do-nothing baseline |

## Q4 — Real clinical video (HyperKvasir; both tests frozen before data)
| | |
|---|---|
| First frozen test (41 clips): polyp clips / controls flagged | **11/21 vs 3/20** — every flagged control held real labeled pathology |
| Its significance (Fisher's exact, two-sided) | **p = 0.020** |
| Miss autopsy: misses traced to sparse frame sampling | **8 of 10** |
| Second frozen test (48 fresh clips): dense vs old protocol, same clips | **19/25 vs 13/25 (76% vs 52%)** |
| Its significance (Fisher's exact on dense's split, 19/25 vs 6/23) | **p = 0.0012** |
| Clean false alarms in test 2 (pre-stated allowance) | **2** — other flagged controls: a labeled cancer, tumor, leak cavity, EMR site |

## Q5 — Why believe any of this
| | |
|---|---|
| Iterations / adoptions / self-caught retractions | **23 / 2 / 2** |
| The two adoptions | **pipeline v2** (in-loop Huber) + **dense-400 protocol** |
| Red-team audit: headline numbers re-derived with independent code | **15/15** (tracking ATEs match to **0.0005 mm**) |
| Full numbers audit from raw artifacts | **35/35 figures reproduce** |

## Deployment (the phone)
| | |
|---|---|
| Phone end-to-end speed vs measured live requirement | **~3 fps vs ~9 needed** → product is the retrospective report card |
| Screen-then-confirm cascade speedup (measured in-app) | **3.2–3.7× faster** screening, teacher-grade findings |

## Five habits when quoting (from the words-to-never-say list)
1. "Poses every frame" — never "never loses track."
2. "Every flagged control held real, LABELED pathology" — never "caught what doctors missed."
3. "A statistical tie; mine poses every frame" — never "beats ORB-SLAM3."
4. "11 of 21 polyp **clips**" — clips, not polyps.
5. "False alarms from **up to** 41%" — the range was 7–41%.
