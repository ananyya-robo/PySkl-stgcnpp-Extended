# Health-class demo (NTU60 XSub, bone stream)

One validation sample per health class (A41-A49), taken with `--sample_idx 0`
(the first validation sample of each class). Illustrative only. Per-class
metrics and calibration are reported separately.

Run from the repo root, with the release unpacked and `data/nturgbd/ntu60_3danno.pkl` in place:

    bash demo/health_class_survey.sh

Checkpoint: `work_dirs/stgcn++/stgcn++_ntu60_xsub_3dkp/b/epoch_16.pth`

| Class | Sample | Predicted | Confidence |
|---|---|---|---|
| A41 sneeze/cough | S001C001P003R001A041 | A41 (correct) | 0.8129 |
| A42 staggering | S001C001P003R001A042 | A42 (correct) | 0.9940 |
| A43 falling | S001C001P003R001A043 | A43 (correct) | 0.9950 |
| A44 touch head | S001C001P003R001A044 | A44 (correct) | 0.9963 |
| A45 touch chest | S001C001P003R001A045 | A45 (correct) | 0.7671 |
| A46 touch back | S001C001P003R001A046 | A46 (correct) | 0.9945 |
| A47 touch neck | S001C001P003R001A047 | A47 (correct) | 0.9556 |
| A48 nausea or vomiting | S001C001P003R001A048 | A48 (correct) | 0.8271 |
| A49 use a fan / feeling warm | S001C001P003R001A049 | A49 (correct) | 0.9982 |

Single-sample confidence differs from the mean confidence reported by the calibration analysis.

To render a sample as a 3D GIF: `python demo/render_sample.py --target_class 42 --out work_dirs/demo_falling_3D.gif`
