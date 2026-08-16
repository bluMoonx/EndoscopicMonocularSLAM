#!/bin/bash
# Temporal inpainting condition: fill glare from neighbor frames (optical-flow warp), then run
# ORB-SLAM3 N times. Appends rows (condition="temporal") to results/severity_experiment.csv.
set +e
cd "C:/Users/moonw/EndoscopicMonocularSLAM"
PY="C:/Users/moonw/anaconda3/envs/slamfair/python.exe"
C=argus_slam
N=5
SEQS="rectified01 rectified04 rectified05 rectified06"
CSV="results/severity_experiment.csv"

for seq in $SEQS; do
  if [ ! -f "data/hamlyn/${seq}_temporal/rgb.txt" ]; then
    echo "[temporal-inpaint] $seq"
    "$PY" speqslam/temporal_inpaint.py --in_seq "data/hamlyn/$seq" --out_seq "data/hamlyn/${seq}_temporal"
  fi
  for run in $(seq 1 $N); do
    out="results/matrix/${seq}_temporal_${run}"
    docker exec $C bash -lc "mkdir -p /workspace/$out && cd /workspace/$out && \
      DISPLAY=:1 /ORB_SLAM3/Examples/Monocular/mono_tum \
      /ORB_SLAM3/Vocabulary/ORBvoc.txt \
      /workspace/configs/hamlyn_${seq}_mono.yaml \
      /workspace/data/hamlyn/${seq}_temporal > run.log 2>&1; true" >/dev/null 2>&1
    kf=$(docker exec $C bash -lc "wc -l < /workspace/$out/KeyFrameTrajectory.txt 2>/dev/null || echo 0" | tr -d '\r ')
    echo "$seq,temporal,$run,$kf" >> "$CSV"
    echo "  $seq temporal run$run -> ${kf} KF"
  done
done
echo "TEMPORAL MATRIX DONE"
