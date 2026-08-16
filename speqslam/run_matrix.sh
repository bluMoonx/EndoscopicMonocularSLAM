#!/bin/bash
# Argus severity experiment: for each sequence, run BASELINE (raw) vs ARGUS (glare-removed)
# N times each, record keyframes. Averaging tames ORB-SLAM3's run-to-run randomness.
# Output: results/severity_experiment.csv  (seq,condition,run,keyframes)
set +e
cd "C:/Users/moonw/EndoscopicMonocularSLAM"
PY="C:/Users/moonw/anaconda3/envs/slamfair/python.exe"
C=argus_slam
N=5
SEQS="rectified01 rectified04 rectified05 rectified06"
CSV="results/severity_experiment.csv"
echo "seq,condition,run,keyframes" > "$CSV"

for seq in $SEQS; do
  # ensure the glare-removed copy exists
  if [ ! -f "data/hamlyn/${seq}_masked/rgb.txt" ]; then
    echo "[preprocess] $seq"
    "$PY" speqslam/glare_removal.py --in_seq "data/hamlyn/$seq" --out_seq "data/hamlyn/${seq}_masked" \
        --method adaptive --k 2.0 --lo 225 --hi 245 --dilate 1
  fi
  for cond in baseline argus; do
    if [ "$cond" = "baseline" ]; then datadir="$seq"; else datadir="${seq}_masked"; fi
    for run in $(seq 1 $N); do
      out="results/matrix/${seq}_${cond}_${run}"
      docker exec $C bash -lc "mkdir -p /workspace/$out && cd /workspace/$out && \
        DISPLAY=:1 /ORB_SLAM3/Examples/Monocular/mono_tum \
        /ORB_SLAM3/Vocabulary/ORBvoc.txt \
        /workspace/configs/hamlyn_${seq}_mono.yaml \
        /workspace/data/hamlyn/${datadir} > run.log 2>&1; true" >/dev/null 2>&1
      kf=$(docker exec $C bash -lc "wc -l < /workspace/$out/KeyFrameTrajectory.txt 2>/dev/null || echo 0" | tr -d '\r ')
      echo "$seq,$cond,$run,$kf" >> "$CSV"
      echo "  $seq $cond run$run -> ${kf} KF"
    done
  done
done
echo "MATRIX DONE -> $CSV"
