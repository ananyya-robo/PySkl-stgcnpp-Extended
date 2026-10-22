#!/usr/bin/env bash
# One validation sample for each health class (A41-A49), NTU60 XSub, bone stream.
# Run from the repo root: bash demo/health_class_survey.sh
# Illustrative only; per-class metrics and calibration are in the main results.
CFG=configs/stgcn++/stgcn++_ntu60_xsub_3dkp/b.py
CKPT=work_dirs/stgcn++/stgcn++_ntu60_xsub_3dkp/b/epoch_16.pth
for c in 42 40 41 43 44 45 46 47 48; do
  echo "=== class $c ==="
  python demo/demo_predict.py --config $CFG --checkpoint $CKPT --target_class $c 2>/dev/null | grep -E "Sample|True class|Predicted class|Confidence|Correct"
done
