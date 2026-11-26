# Argus: A 3D Domain-Adapted Monocular Depth and Odometry Pipeline for Blind-Spot Quantification in Colonoscopy

**GSDSEF Digital Notebook — working copy.** The live editable version is a Claude doc; the final
deliverable is a PDF exported from it (zFairs upload due Feb 26, 2027). This file mirrors the draft
so the repo carries the notebook's text. *[INSERT IMAGE: ...]* markers show where screenshots and
figures go before export (checklist in Appendix C).

---

## Abstract

Colonoscopy misses roughly one in four precancerous polyps, largely because parts of the colon wall are never brought into view and no standard instrument tells the endoscopist which parts those were. This project developed Argus, a system that answers that question using only the video of a standard, unmodified endoscope. The pipeline chains a fine-tuned monocular depth network, a visual odometry tracker written from scratch, 3D surface fusion, per-patch view counting, and a cascaded polyp detector, and the whole system runs on one workstation and an ordinary Android phone. Development went through 23 design iterations, each judged against pass or fail criteria committed to a public git record before the test ran; only 2 passed their gates and were adopted. Against phantom ground truth, the final pipeline tracks the camera to 0.42 mm mean error and reconstructs the surface to about 2 mm, and fine-tuning the depth network cut false blind-spot alarms from up to 41% of the wall to under 4%. The same pattern repeated in detection, where a redesigned sampling protocol raised detection on unseen clinical footage from 13 of 25 to 19 of 25 polyp clips. Head to head, the tracker ties the standard tool ORB-SLAM3 on accuracy while posing every frame of video. These results show that inspection coverage can be measured from the video a doctor already has, and that generic AI fails inside the body until it is specifically adapted to it.

## Acknowledgements

I would like to thank my parents for their support and patience through a year of long nights on this project.

I thank Dr. Aasma Shaukat (NYU Langone, lead author of the current ACG colorectal cancer screening guideline) and Dr. Douglas Rex (Indiana University) for replying to my questions about whether an objective measure of mucosal exposure could serve as a colonoscopy quality indicator. Both gave permission to be mentioned here. Dr. Shaukat identified mucosal inspection as "a vital quality issue" and "a target for improvement with AI," and recommended validation on publicly available colonoscopy videos as the next step, which this project carried out.

I also thank the research groups who published the open datasets this project is validated on: the C3VD colonoscopy phantom dataset, the EndoSLAM dataset, the Hamlyn Centre endoscopic video datasets, and the HyperKvasir and Kvasir-SEG clinical datasets.

**AI-use disclosure:** I used an AI assistant (Anthropic's Claude) as an engineering collaborator for code drafting and analysis under my direction. All experiments ran on my own hardware against pass or fail criteria committed in advance, every published number is reproducible from the repository's raw artifacts, and the responsibility for all of it is mine. This project used no institutional lab and no mentorship program.

## Introduction

Colonoscopy is the main tool for preventing colorectal cancer. A doctor guides a camera through the colon, looks for small growths called polyps, and removes the ones that could become cancer. The procedure works, but it is not complete: tandem studies estimate that about one in four precancerous polyps is missed [3]. A major reason is simple. Parts of the colon wall never appear on screen at all. They hide behind folds, pass by too quickly, or are seen only at a glancing angle, and the endoscopist has no record of which parts those were.

This project began with a narrower idea. Endoscopic video is full of bright lens glare, and my original plan was to remove that glare so that camera-tracking software would work better inside the body. Testing mostly disproved that idea, but the work left behind something more useful: a camera tracker that worked on colon video. That raised a bigger question. If software can track where the camera was and estimate how far away the wall is in every frame, then it can also count how many times each patch of wall was actually seen, and mark the patches that never were.

The goal of this project became exactly that: build a system that takes the ordinary video from a standard, unmodified endoscope and produces a 3D map of the colon with the under-inspected regions marked, using no depth sensor, no external tracking, and no new hardware. The system is named Argus, and this notebook documents how it was built, how it was tested, where it succeeds, and where it provably cannot go.

## Review of Literature

This project sits at the intersection of two fields: colonoscopy quality research, which defines the problem, and computer vision, which supplies the tools. This review covers both, and ends with the gap this project was built to fill.

### Colorectal cancer, polyps, and the miss rate

Most colorectal cancers develop from adenomas, small precancerous growths on the colon wall. Removing an adenoma during a screening colonoscopy prevents the cancer it could have become, which is why screening guidelines recommend regular colonoscopies starting at age 45 [1]. The procedure's protective power depends entirely on finding what is there. A 2019 meta-analysis of tandem colonoscopy studies, where a second colonoscopy immediately follows the first and counts what the first one missed, estimated that about 26% of adenomas are missed [3]. The miss mechanisms reported in that literature are largely geometric: lesions hidden behind haustral folds, located behind flexures, or passed during fast withdrawal. In other words, a large share of missed polyps were never on screen in the first place.

### How colonoscopy quality is measured today

Current quality metrics are statistical proxies for the doctor rather than measurements of the procedure. The main one is the adenoma detection rate (ADR), the fraction of a doctor's screening procedures that find at least one adenoma. ADR matters: a large study of over 300,000 procedures found that each 1% increase in a doctor's ADR was associated with a 3% decrease in their patients' risk of interval colorectal cancer [2]. The other common metric is withdrawal time, with at least six minutes recommended for the inspection phase [1]. Both metrics describe an endoscopist's average behavior. Neither says anything about an individual procedure, and neither measures the quantity that the miss-rate literature points at: how much of this patient's colon wall was actually inspected. No standard instrument measures that today.

### AI in colonoscopy: detection, not coverage

The most mature AI in colonoscopy is computer-aided detection (CADe), which draws a box or outline around polyps that appear on screen in real time. Modern CADe systems are built on semantic segmentation architectures such as DeepLabV3+ [9], and are trained and benchmarked on public datasets of labeled clinical footage such as Kvasir-SEG and HyperKvasir, a large multi-class image and video dataset collected from real procedures in Norway [10]. CADe addresses a different failure than this project does. By construction, a detector can only analyze frames that exist; it cannot account for wall that never appears on screen. Detection and coverage are therefore complementary questions, and this project includes a detector only as one pipeline stage, not as its purpose.

### Camera tracking: SLAM and visual odometry

Estimating where a camera is from its own video is a mature field called SLAM (simultaneous localization and mapping). The standard open-source system is ORB-SLAM3, which detects distinctive feature points in each image, matches them across frames, and maintains a map of landmarks, storing camera poses at selected snapshot frames called keyframes [7]. An alternative family, direct methods such as Direct Sparse Odometry, skips feature points and estimates motion by minimizing the photometric (brightness) difference between frames directly [8]. Direct methods suit low-texture scenes, which matters here: the colon is a wet, smooth, repetitively folded tube, lit by a single light that moves with the camera. It offers few of the stable corners and textures that feature-based systems were designed around. The tracker built in this project is a direct method, estimating full 6-degree-of-freedom camera motion by Gauss-Newton optimization of the photometric error, with the specific requirement that it output a pose for every single frame, because coverage counting cannot tolerate gaps.

### Monocular depth estimation and the domain gap

A single camera does not measure distance, so depth must be inferred. Modern monocular depth networks such as Depth Anything V2 are trained on tens of millions of ordinary photographs and produce dense per-pixel depth estimates from one image [6]. Their published strength is generality across everyday scenes. The inside of a colon, however, is unlike any everyday scene: there is no sky, no floor, no familiar object scale, the lighting comes from the camera itself, and the surfaces are wet and specular. Whether such a network's competence extends into the body is exactly the kind of domain-gap question that this project ended up measuring, in two different components.

### Validation data for endoscopic 3D

Accuracy claims need ground truth, and inside a living patient there is none: no sensor records where the endoscope tip truly was. The field solves this with phantoms and instrumented recordings. The C3VD dataset provides colonoscopy video of a silicone colon phantom filmed with a real endoscope whose full 6-DOF pose was recorded by registration, along with the phantom's true 3D geometry, making it the only public data where trajectory error and surface error can be measured in millimeters [4]. The EndoSLAM dataset provides endoscopic video of ex-vivo porcine (pig) organs with recorded camera poses, allowing cross-dataset tests on real tissue [5]. For clinical realism without geometric ground truth, HyperKvasir provides real procedure footage with expert content labels [10]. Comparing an estimated camera path against a recorded true path requires aligning the two trajectories first; the standard least-squares alignment method is Umeyama's algorithm [11], and the aligned error is reported as absolute trajectory error (ATE).

### The gap this project addresses

Taken together, the literature establishes three things: a quarter of adenomas are missed and the mechanism is largely unseen wall [3]; the field's quality metrics do not measure per-procedure coverage [1][2]; and the component technologies, direct tracking [8], monocular depth [6], and dense fusion, exist but are built and benchmarked on everyday scenes rather than inside bodies. What did not exist was a validated system that chains these components into a coverage measurement from standard monocular colonoscopy video alone, with its accuracy measured against ground truth and its failure modes characterized. Building and honestly testing that system is this project.

## Statement of the Problem

Can the fraction of colon wall that was actually inspected during a colonoscopy be measured from the video of a standard, unmodified endoscope alone, with no depth sensor, no external tracking, and no new hardware, accurately enough to flag under-inspected regions?

## Hypotheses

This project had two hypotheses in sequence, because the first one was tested and largely rejected, and the investigation it opened produced the second.

**Hypothesis 1 (original):** Specular glare (bright lens reflections) is a major cause of camera-tracking failure on endoscopic video, so detecting and removing glare will measurably improve tracking accuracy.

**Hypothesis 2 (revised, after Hypothesis 1 was largely disproven):** A pipeline that chains (a) a monocular depth network fine-tuned on colonoscopy data, (b) a direct visual odometry tracker that outputs a camera pose for every frame, (c) 3D surface fusion, and (d) per-patch view counting can produce a blind-spot map of the colon from monocular video alone, with camera tracking accurate to within a few millimeters of recorded ground truth and a false blind-spot rate low enough for the map to be meaningful (under 5% of wall falsely flagged).

Both hypotheses are testable against the C3VD phantom, the only public data with recorded camera-pose and surface ground truth. Every quantitative pass or fail criterion used to judge them was committed to the public git repository before the corresponding test ran.

## Materials

**Hardware (all consumer-grade; no institutional lab or cloud compute was used):**

- One Windows workstation with a single NVIDIA RTX A1000 GPU (8 GB VRAM). All neural-network fine-tuning ran on this card (about 40 minutes per training fold).
- One ordinary Android phone, used to deploy and benchmark the complete pipeline on-device.
- One laptop for fair-day demonstration (runs the same repository).

**Datasets (all public):**

- C3VD: colonoscopy video of a silicone colon phantom with recorded 6-DOF camera poses and known 3D geometry [4]. The only data with trajectory and surface ground truth; used for all accuracy validation.
- C3VDv2: deforming-phantom sequences from the same group, used to measure the cost of tissue deformation against a rigid control.
- EndoSLAM: ex-vivo porcine colon video with recorded camera poses [5]; used for the cross-dataset tracker comparison on real tissue.
- Hamlyn Centre in-vivo endoscopic video; used for qualitative live demonstrations (no pose ground truth exists for it).
- HyperKvasir: real clinical colonoscopy videos with expert content labels [10]; used for the two frozen detection tests (41 and 48 clips).
- Kvasir-SEG: expert-segmented polyp images; used for detector training and the demo gallery.

**Software:**

- Python 3.10 with PyTorch, OpenCV, and the Hugging Face transformers library (workstation pipeline, training, and evaluation).
- Depth Anything V2 (small) as the depth backbone [6]; DeepLabV3+ with an EfficientNet-B4 encoder for polyp segmentation [9].
- ORB-SLAM3 as the published baseline tracker for head-to-head comparisons [7].
- Flask for Argus Studio, a local web console that runs every demo and experiment station on one page.
- Android Studio with Kotlin and Jetpack Compose for Argus Lite, the phone app; models quantized to int8 for on-device inference.
- Git and GitHub for the public record: all code, pre-registered pass/fail gates, iteration history, and results artifacts live in one repository (github.com/bluMoonx/EndoscopicMonocularSLAM).

*[INSERT IMAGE: photo of the workstation and phone setup]*

## Procedures

The project ran in six phases over four months (July to November 2026). Every date below is fixed by a public git commit.

**PHASE 0: The glare study (July 13 to August 15). Hypothesis 1.**

1. Built a specular-glare detector (brightness thresholding with dilation) and verified it visually on real in-vivo frames.
2. Ran ORB-SLAM3 on endoscopic sequences with and without glare masking, and with two inpainting methods (AI inpainting and temporal inpainting) that fill the glare regions with plausible tissue.
3. Compared tracking accuracy across conditions. Result: ORB-SLAM3's internal outlier rejection (RANSAC) already absorbs most glare damage, and naive inpainting made tracking worse. Hypothesis 1 was rejected with a mechanism, and the project pivoted.

**PHASE 1: Building the four-stage pipeline (August 15 to September 16). Hypothesis 2, construction.**

4. Wrote a direct visual odometry tracker from scratch: for each pair of frames, estimate the 6-DOF camera motion by Gauss-Newton minimization of the photometric error, using the depth estimate to relate pixel motion to physical motion. Required property: a pose for every frame, no keyframe gaps.
5. Added a divergence guard that flags any frame pair the optimizer cannot solve, so failure is visible and countable instead of silent.
6. Fine-tuned the Depth Anything V2 depth network on C3VD phantom frames, using leave-one-sequence-out training so every evaluation sequence is unseen by its model.
7. Fused the per-frame depth maps and camera poses into one 3D surface using TSDF (truncated signed distance function) fusion at 0.6 mm voxels.
8. Counted, for every patch of the fused surface, how many frames gave a usable view of it (in frame, unoccluded, within range). Marked patches with fewer than 3 views as blind spots.
9. Added the detection stage: a DeepLabV3+ polyp segmenter, later arranged as a screen-then-confirm cascade (a fast student screens every frame, the full-size teacher confirms anything suspicious).

**PHASE 2: Validation against ground truth (September, ongoing).**

10. Measured tracking error (ATE after Umeyama alignment [11]) against the phantom's recorded camera paths, and surface error (Chamfer distance) against the phantom's known geometry. Printed every trajectory number beside a constant-point baseline (a fake tracker that says the camera never moved), so a flattering but meaningless number cannot hide.
11. Ran the staged ablation: re-ran the pipeline repeatedly, each time replacing one stage's estimated input with its ground-truth version, to isolate which stage caused the blind-spot errors. The ablation convicted the depth network and cleared the tracker.
12. Applied exactly one fix where the ablation pointed: domain fine-tuning of the depth network (step 6), then re-measured everything.

**PHASE 3: The reboot into frozen protocols (September 16 to 26).**

13. Caught and retracted a published full-withdrawal tracking figure (150 mm) after an audit showed the tracker had silently diverged on 24 frame pairs and trajectory alignment had dressed the result up (a do-nothing baseline scored 168 mm on the same data). Made the guard-rejection rate a primary metric.
14. Adopted the pre-registration rule for all further work: every experiment's pass or fail criteria are committed to the public repository before the test runs, failures are published with cause-of-death analyses, and frozen test data is read exactly once.
15. Re-derived all 35 published figures from raw artifacts in an automated audit. All 35 reproduced.

**PHASE 4: Pre-registered campaigns (September 26 to October 17).**

16. Loop closure: instead of building a loop-closure system, granted the pipeline perfect loop closures from ground truth (an oracle), measuring the ceiling any real implementation could reach.
17. Deformation: measured the tracking cost of deforming tissue on C3VDv2 against a speed-matched rigid control, then designed six candidate fixes, each with a pass bar committed in advance, and tested all six.
18. Frame rate: measured tracking on complete expert withdrawals at four frame-sampling rates to find where whole-colon tracking fails and where it holds.
19. Detection, first frozen test: froze the detector and thresholds, committed them, then downloaded 41 unseen HyperKvasir clips (21 polyp, 20 control) and scored one pass. Autopsied every miss.
20. Detection, second frozen test: the autopsy pointed at sparse frame sampling, so a dense-sampling protocol was pre-registered with three gates and tested once on 48 fresh clips.
21. Pipeline v2: a robust in-loop Huber weighting in the tracker, adopted only after a pre-registered four-stage re-validation of every tracking benchmark. Also tested whether the upgrade transfers to the phone's tracker (it does not; the phone kept its own kernel).

**PHASE 5: Head-to-head against the standard tool (October 17 onward).**

22. Pre-registered and ran two comparisons against ORB-SLAM3: on the C3VD phantom, and on EndoSLAM ex-vivo porcine colon, with decision rules committed before either system ran.

**PHASE 6: Deployment and independent audit (October to November).**

23. Built Argus Lite: the full pipeline (quantized depth, guarded tracking, fusion, cascade detection, 3D map viewer with blind-spot overlay) running on an ordinary Android phone, including a live benchmark card that re-measures the models on whatever device it runs on.
24. Built Argus Studio: a local web console that serves every demo and experiment station from one page, fully offline.
25. Wrote a red-team audit that re-derives every headline number with independently written code, including a different trajectory alignment algorithm (Horn instead of Umeyama), fresh parsers, and an end-to-end re-track.

*[INSERT IMAGE: pipeline block diagram]*

*[INSERT IMAGE: project timeline figure]*

## Results

Every accuracy number below was measured against C3VD phantom ground truth unless noted otherwise, and every trajectory error is published beside its constant-point (do-nothing) baseline.

**Table 1. Final pipeline accuracy vs ground truth.**

| Measurement | Result |
| --- | --- |
| Camera tracking error, mean ATE (pipeline v2, 5 sequences) | 0.42 mm |
| Camera tracking error before the v2 adoption (and the phone's own tracker today) | 0.91 mm |
| 3D surface error, median Chamfer (4 of 5 sequences) | 1.3 to 2.4 mm |
| Blind-spot agreement with a perfect-tracking twin instrument | 98.6% |
| Monocular sizing error on 3 to 30 mm targets, median | 0.60 mm |

**Table 2. The depth fine-tuning fix (the staged ablation's one intervention).**

| Metric | Off-the-shelf depth | Fine-tuned depth |
| --- | --- | --- |
| Depth accuracy (pixels within 25% of truth, held-out) | 68% | 92 to 99.7% |
| Wall falsely flagged as blind | 7 to 41% (by sequence) | 0.5 to 3.9% |

*[INSERT IMAGE: ablation bar chart, results/ folder]*

*[INSERT IMAGE: depth X-ray strip, off-the-shelf vs fine-tuned]*

**Table 3. The frame-rate requirement (complete expert withdrawals, replicated on three).**

| Effective frame rate | Guard-rejection rate | Verdict |
| --- | --- | --- |
| ~2 fps | 16 to 23% | tracking fails |
| ~9 fps | 1.5 to 4.2% | tracking holds |
| native rate | 0.7 to 2.1% | solved |

Real endoscopes stream 25 or more frames per second, so the requirement is satisfiable; the burden moves to processing speed. On the flagship full withdrawal (3,667 frames analyzed, 7,003 mm of camera path), the path error was 92 mm (1.3%) where the do-nothing baseline scores 133 mm. At screening pace, about 40% of the mapped wall was genuinely under-seen, confirmed by a twin instrument driven by the recorded true camera path.

**Table 4. Head-to-head against ORB-SLAM3 (both pre-registered).**

| Test | Argus | ORB-SLAM3 | Verdict |
| --- | --- | --- | --- |
| C3VD phantom, mean ATE | 0.42 mm | 0.39 mm | a draw (per-sequence 2-2-1, confidence intervals overlap) |
| EndoSLAM ex-vivo porcine colon, mean ATE | 39.6 to 41.4 mm | 39.6 to 41.4 mm | four-way statistical tie across all variants |
| Fraction of each sequence posed | every frame, zero loss | 65 to 95%, sparse keyframes | the property coverage counting requires |

**Table 5. Frozen tests on unseen clinical video (HyperKvasir [10]).**

| Test | Result | Significance |
| --- | --- | --- |
| First frozen test, 41 clips | 11/21 polyp clips flagged vs 3/20 controls; every flagged control contained real labeled pathology | Fisher's exact p = 0.020 |
| Miss autopsy | 8 of 10 misses traced to sparse frame sampling | led to one pre-registered fix |
| Second frozen test, 48 fresh clips | dense protocol 19/25 polyp clips vs 13/25 for the old protocol on identical clips; 2 clean false alarms (within the pre-stated allowance); other flagged controls held a labeled cancer, tumor, leak cavity, and EMR site | Fisher's exact p = 0.0012; dense protocol adopted |

**Table 6. Measured limits (each with its mechanism; see Data Analysis).**

| Limit | Measurement |
| --- | --- |
| Whole-colon drift over a full withdrawal | ~100 mm |
| Drift recoverable by perfect (oracle) loop closure | at most 14% |
| Tracking cost of deforming tissue | ~9x |
| Deformation fixes attempted / passed their pre-registered bars | 6 / 0 (best miss: minus 18% against a +20% bar) |
| Cross-session site re-identification vs clinical need | ~26 mm achieved vs ~10 mm needed; closed as out of reach |
| Phone end-to-end speed vs measured live requirement | ~3 fps vs ~9 fps; shipped product is the retrospective report card |

**Process record:** 23 pre-registered iterations, 2 adoptions (pipeline v2 and the dense sampling protocol), 2 self-caught retractions. A red-team audit re-derived all 15 headline numbers with independently written code; tracking ATEs matched to 0.0005 mm and a fresh end-to-end re-track reproduced the 0.42 mm mean exactly. A separate 35-figure audit reproduces every published figure from raw artifacts.

*[INSERT IMAGE: app 3D map with blind-spot overlay]*

*[INSERT IMAGE: withdrawal-speed curve (cost of rushing)]*

*[INSERT IMAGE: procedure report card]*

## Data Analysis

**1. The staged ablation isolated the bottleneck instead of guessing at it.**

Swapping each estimated input for its ground-truth version, one stage at a time, showed that handing the system perfect camera poses barely changed the blind-spot map, while handing it perfect depth collapsed the error. That cleared the tracker and convicted the depth network. The network had been trained on everyday photographs; a colon is a wet tube lit by a light attached to the camera, and the network's distance estimates there were badly wrong (68% accuracy, up to 41% of the wall falsely flagged blind). One targeted fix, fine-tuning on colonoscopy frames, moved depth accuracy to 92 to 99.7% and false alarms under 4% (Table 2). The same disease then appeared a second time in a different component: the detection stage performed poorly on real clinical footage until its sampling protocol was adapted to how polyps actually appear in real clips (Table 5). Demonstrated twice in two components, this is the project's central finding: the competence of a general-purpose AI model ends at the edge of its training data, and targeted adaptation can rescue it.

**2. Why a from-scratch tracker survives a tie.**

The head-to-heads (Table 4) show the custom tracker is not more accurate than ORB-SLAM3, and this project does not claim it is. The reason it exists is what it outputs. ORB-SLAM3 saves effort by posing only selected keyframes, and on these videos it also loses tracking entirely for stretches, covering 65 to 95% of a sequence. Coverage counting needs a camera pose for every single frame, because a patch of wall may be visible only in the frames a keyframe system skipped. On real tissue, the four-way tie at 39.6 to 41.4 mm across every tracker and variant indicates the error floor there comes from the data (deformation, lighting), not from any tracker's design.

**3. The limits are mechanisms, not excuses.**

Drift: small per-frame errors accumulate to ~100 mm over a full withdrawal. The oracle experiment granted the system perfect loop closures and recovered at most 14%, because loop closure needs the camera path to cross itself and a withdrawal is topologically a line. The drift is therefore a property of the task, not a software bug. Deformation: a moving colon violates the pipeline's rigid-scene assumption at a measured cost of about 9x, and six pre-registered fixes all failed their own bars, the best missing a +20% target at minus 18%. These two limits together killed one intended application: re-finding a marked site in a later procedure landed at ~26 mm against a ~10 mm clinical need, so that goal was closed with the number that killed it. The frame-rate limit, by contrast, turned out to be solvable: tracking fails at ~2 effective fps and holds from ~9, and real endoscopes stream 25 or more.

**4. The statistics of the clinical tests.**

Both frozen tests used Fisher's exact test on the 2x2 table of clip type (polyp / control) versus flagged (yes / no), because the small cell counts rule out a normal-approximation two-proportion test. The first test's split (11/21 vs 3/20) gives p = 0.020; the dense protocol's split (19/25 vs 6/23) gives p = 0.0012. The dense-versus-old comparison (19 vs 13) was a matched-pairs design: both protocols scored identical clips, so clip difficulty cancels out, and the pre-registered criterion was simply that dense must beat old. Critically, the thresholds and gates were committed before any test clip was downloaded, which is what makes these p-values honest: no test was re-run until significant. The three flagged controls in the first test all contained pathology already labeled in the dataset (not findings the doctors missed), so the false-alarm rate on genuinely clean tissue was effectively zero.

**5. Honest sources of error and scope.**

All millimeter accuracy claims rest on a rigid phantom, because that is the only place trajectory ground truth exists; generalization to diverse living human colons is unproven, and the data to prove it does not publicly exist. Coverage measures exposure, not inspection: a patch counted as seen was on screen, which is necessary but not sufficient for a doctor to have examined it. On the phone, distances from uncalibrated cameras are labeled estimates. Each number in this notebook is reported beside the baseline or caveat that keeps it honest.

## Conclusion

The question was whether inspection coverage can be measured from the video of a standard, unmodified endoscope alone. The answer is yes, within characterized limits.

Hypothesis 1 was not supported. Glare was not the main obstacle to tracking on endoscopic video: the standard tracker's own outlier rejection already absorbs most of it, and naive glare removal made tracking worse. Publishing that rejection, with its mechanism, is part of this project's record.

Hypothesis 2 was supported. The four-stage pipeline tracks the camera to 0.42 mm mean error against recorded ground truth, reconstructs the colon surface to about 2 mm from video alone, and agrees with a perfect-tracking twin instrument on 98.6% of blind-spot labels. The false blind-spot rate ended under 4% of wall, inside the pre-stated 5% bar, but only after domain fine-tuning; off the shelf, the depth network falsely flagged up to 41%. At realistic screening pace the instrument measures that roughly 40% of the mapped wall is genuinely under-seen, which is consistent with what the miss-rate literature implies but which no current quality metric measures directly.

The explanation that ties the whole project together is the domain gap. A general-purpose depth network failed inside the body and was rescued by targeted fine-tuning; a general-purpose detection setup failed on real clinical footage and was rescued by a protocol adapted to it. Both rescues were pre-registered and tested on frozen data. The broader lesson is that AI competence ends at the edge of the training data, and that inside the human body, adaptation is not optional.

Equally important is what this project proved it cannot do. Whole-colon drift of ~100 mm is a property of the task, since a withdrawal path is a line and even perfect loop closure recovers at most 14%. Tissue deformation costs about 9x in accuracy and survived six pre-registered fix attempts. Those two limits close cross-session site relocation (~26 mm against a ~10 mm need) as out of reach for this design. Each limitation ships with its mechanism.

## Recommendations

**Possible applications of the findings:**

1. A per-procedure quality report. The most direct use is the retrospective report card Argus already produces: coverage grade, blind regions in square millimeters, and camera speed, generated from the video a clinic already records. This would measure the individual procedure, complementing ADR and withdrawal time, which only describe the doctor on average. In correspondence, the lead author of the current ACG screening guideline wrote that a reliably measured and validated coverage metric could change quality metrics, and recommended validation on public videos (done in this project) followed by a pilot study with patients.
2. Training feedback. A trainee endoscopist could review their own withdrawals with the blind-spot map and see exactly which wall regions they tend to pass, something no current feedback tool shows.
3. Live guidance, eventually. The measured requirement is ~9 fps; the phone currently reaches ~3. Closing that gap would let the system flag under-seen regions before withdrawal ends, so the doctor could go back while the scope is still in.

**Additional testing and improvements:**

1. Real-tissue validation of coverage. All millimeter claims rest on a rigid phantom. The decisive next experiment needs living, deforming colon with some form of position ground truth, data that does not publicly exist today.
2. Processing speed. The depth network is the measured bottleneck on-device; resolution reduction was tested and rejected under its gates, so the lever is likely model architecture or dedicated hardware.
3. Deformation remains open. Six fixes failed their bars, each with a mechanism; those mechanisms (especially that wall motion toward the camera is projectively invisible to one camera) say where not to look, which is a map for whoever tries next.
4. A clinical pilot, per the guideline author's suggested path, once coverage is validated on more diverse footage.

**Reflection and continuation.** I would continue this project. One year was enough to build the four-stage instrument, validate it against ground truth, and establish its limits honestly. The methodology, pre-registering every pass or fail bar in public git before each test, is the part I will carry into everything I build next, because it is the reason I can defend every number in this notebook.

## Bibliography

[1] Shaukat A. et al. ACG Clinical Guidelines: Colorectal Cancer Screening 2021. American Journal of Gastroenterology 116(3), 2021.

[2] Corley D.A. et al. Adenoma Detection Rate and Risk of Colorectal Cancer and Death. New England Journal of Medicine 370, 2014.

[3] Zhao S. et al. Magnitude, Risk Factors, and Factors Associated With Adenoma Miss Rate of Tandem Colonoscopy. Gastroenterology 156(6), 2019.

[4] Bobrow T.L. et al. Colonoscopy 3D Video Dataset with Paired Depth from 2D-3D Registration. Medical Image Analysis, 2023.

[5] Ozyoruk K.B. et al. EndoSLAM Dataset and an Unsupervised Monocular Visual Odometry and Depth Estimation Approach for Endoscopic Videos. Medical Image Analysis 71, 2021.

[6] Yang L. et al. Depth Anything V2. Advances in Neural Information Processing Systems (NeurIPS), 2024.

[7] Campos C. et al. ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM. IEEE Transactions on Robotics 37(6), 2021.

[8] Engel J. et al. Direct Sparse Odometry. IEEE Transactions on Pattern Analysis and Machine Intelligence 40(3), 2018.

[9] Chen L.-C. et al. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. European Conference on Computer Vision (ECCV), 2018.

[10] Borgli H. et al. HyperKvasir, a Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy. Scientific Data 7, 2020.

[11] Umeyama S. Least-Squares Estimation of Transformation Parameters Between Two Point Patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence 13(4), 1991.

Correspondence: Dr. Aasma Shaukat (NYU Langone) and Dr. Douglas Rex (Indiana University), email correspondence on mucosal exposure as a quality indicator, November 2026 (quoted with permission).

## Appendix

### A. Daily log

Every entry below is anchored to a public git commit in the project repository, so each date is verifiable, not reconstructed from memory.

| Date (2026) | Entry |
| --- | --- |
| Jul 13 | Repository created. |
| Aug 15 | Act I committed: glare detector, masking and inpainting tests. Hypothesis 1 rejected with a mechanism (ORB-SLAM3's RANSAC already absorbs glare; inpainting hurts). Pivot to building a direct tracker. |
| Aug 15 - Sep 16 | Act II, construction: from-scratch direct VO, divergence guard, depth fine-tuning (leave-one-sequence-out), TSDF fusion, view counting, detection stage, first phone deployment. Staged ablation convicts depth; fine-tuning fix measured (Table 2). |
| Sep 16 | Pre-registration rule adopted: pass/fail criteria committed before each test, failures published with autopsies. |
| Sep 19 | Deformation cost measured on deforming phantoms: ~9x vs speed-matched rigid control. |
| Sep 20 | First full-withdrawal figure (150 mm) retracted after self-audit: silent divergence on 24 frame pairs, alignment flattering the result (do-nothing baseline 168 mm). Guard rate made a primary metric. |
| Sep 26 | 35-figure numbers audit passes; pre-registration rule hardened (frozen tests read once). |
| Oct 1 | Detection cascade adopted under its gates (student screens 3.2 to 3.7x faster, teacher confirms). |
| Oct 4 | Deformation fixes 2 and 3 (in-loop Huber as a deformation fix, oscillation smoothing) both fail their bars. Fix 4 (explicit wall-displacement model) fails at +18% vs a +20% bar. Fix 5 (adaptive gated field) dead at development. |
| Oct 5 - 6 | Deformation fix 6 (learned correspondence weighting) fails its frozen verdict. Total: six attempts, six rejections, each with a mechanism. Depth speed push: both resolution candidates rejected. |
| Oct 8 | Pipeline v2 (in-loop Huber as a tracker upgrade) adopted after a pre-registered four-stage re-validation; headline mean 0.91 to 0.42 mm. |
| Oct 10 | Huber app-port trial fails one of two pre-registered windows; the phone keeps its own kernel. Adoption validated per configuration, never assumed to generalize. |
| Oct 11 | Dense-400 screening protocol adopted: second frozen test on 48 fresh clips passes all three gates (19/25 vs 13/25, p = 0.0012). |
| Oct 17 | Iteration record closed at 23 iterations, 2 adoptions, 2 retractions. ORB-SLAM3 phantom head-to-head published: a draw. |
| Late Oct | EndoSLAM cross-dataset head-to-head published: four-way statistical tie on real tissue. Red-team audit: 15/15 headline numbers re-derived with independent code. |
| Nov 2 - 3 | Correspondence with Dr. Rex and Dr. Shaukat; both reply. Permission to quote granted. |
| Nov | GSDSEF registration; digital presentation and this notebook assembled. |

### B. Raw data and reproducibility

All raw data lives in the public repository rather than printed here, because every table in this notebook is regenerated from it by script:

- ITERATION_HISTORY.md: the full 23-iteration ledger with commit hashes.
- results/: per-sequence trajectory files, ablation and robustness CSVs, the frozen clinical test manifests and verdicts (dense_protocol_gate.md, dense_t3_verdict.json), report cards, and 3D model viewers.
- Pre-registered gates: results/orbslam3_baseline_gate.md, results/orbslam3_endoslam_gate.md, results/dense_protocol_gate.md, results/huber_port_gate.md, and the deformation campaign bars, each committed before its test ran.
- speqslam/audit_numbers.py: re-derives all 35 published figures from raw artifacts.
- speqslam/redteam_audit.py: re-derives the 15 headline numbers with independently written code (Horn alignment instead of Umeyama, fresh parsers, end-to-end re-track).

### C. Figures to include

*Checklist of images to paste into the marked placeholders before export: workstation photo; pipeline block diagram; project timeline; ablation bar chart; depth X-ray strip; app 3D map with blind-spot overlay; withdrawal-speed curve; a procedure report card; gate-commit screenshot (pass/fail criteria in git before the run); Argus Studio home screen.*
