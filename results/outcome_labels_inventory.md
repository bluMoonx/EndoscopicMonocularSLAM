# Data inventory: can coverage metrics be tested against clinical outcomes? (Oct 17 2026)

Question (from correspondence with the C3VD group): coverage metrics matter insofar as
they stratify procedures against outcomes such as adenoma detection rate, interval
cancers, or post-polypectomy site re-identification. Is any public dataset sufficient
to attempt that correlation?

Finding: NO public colonoscopy video dataset ships per-procedure outcome or quality
labels. On-disk check: HyperKvasir's video annotations are finding labels (polyp,
cancer, "nothing", procedures), with no withdrawal-time, prep-quality or outcome field
per video. Literature check: withdrawal-time annotations exist in private research
collections (e.g. Katzir et al. 2022, 788 videos, not released); Endomapper carries
anatomical segment labels on a handful of cases; ADR is computed from clinical records,
not video, and no release ties it to procedure footage.

Consequence: the coverage-vs-outcome study is a retrospective CLINICAL study, which is
precisely the work the dataset's authors describe pursuing with IRB-governed data. For
this project the wall is documented, named in future work, and left honestly unclimbed:
the same category of limitation as in-vivo pose ground truth.
