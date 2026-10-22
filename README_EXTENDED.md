# PySkl-stgcnpp-Extended
MSc dissertation work built on PYSKL (Duan et al., 2022, Apache-2.0): https://github.com/kennymckormick/pyskl

## Added by me
- `tools/fuse_2s_*.py`, `tools/fuse_4s_*.py`: stream fusion and metrics
- `tools/calibration.py`: ECE, Brier score, mean confidence, reliability diagrams
- `tools/check_score_format.py`, `demo/make_skeleton_gif.py`, `demo/demo_predict.py`
- Trained checkpoints and score files: see the `v1-trained` release
- `demo/render_sample.py`: renders one validation sample of a chosen class to a 3D skeleton GIF
- `demo/health_class_survey.sh`: runs `demo_predict.py` on one validation sample for each health class (A41-A49)
- `demo/HEALTH_CLASS_DEMO.md`: results of that survey (NTU60 XSub, bone stream)
- `results/`: reliability diagrams (`calibration/`) and single-stream ROC curves (`roc/`) from `calibration.py` and `tools/test.py`
## Inherited with the project copy (edited by me where noted)
- `tools/test.py` (single-stream precision/recall/AUC/ROC code; I fixed ROC saving)
- `demo/vision_gif.py`

Dataset files are not included; download links are in `tools/data/README.md`.
