import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# NTU RGB+D 25-joint skeleton, 0-indexed (same layout PySKL uses)
EDGES = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
         (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
         (15, 14), (16, 0), (17, 16), (18, 17), (19, 18),
         (21, 22), (22, 7), (23, 24), (24, 11)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_pkl', default='data/nturgbd/ntu60_3danno.pkl')
    ap.add_argument('--split', default='xsub_val')
    ap.add_argument('--target_class', type=int, required=True)
    ap.add_argument('--sample_idx', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.dataset_pkl, 'rb') as f:
        data = pickle.load(f)

    # identical selection logic to demo_predict.py, so the sample matches
    val_names = set(data['split'][args.split])
    matches = [a for a in data['annotations']
               if a['frame_dir'] in val_names and a['label'] == args.target_class]
    ann = matches[args.sample_idx]

    kp = np.asarray(ann['keypoint'])            # (M, T, V, 3)
    M, T, V, _ = kp.shape
    valid = [b for b in range(M) if np.abs(kp[b]).sum() > 0]

    pts = np.concatenate([kp[b].reshape(-1, 3) for b in valid])
    pts = pts[np.abs(pts).sum(1) > 0]
    lo, hi = pts.min(0), pts.max(0)
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    def draw(t):
        ax.cla()
        # Kinect: y is up, z is depth -> plot depth on the ground plane
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[2] - half, centre[2] + half)
        ax.set_zlim(centre[1] - half, centre[1] + half)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=15, azim=-60)
        for b in valid:
            j = kp[b, t]
            if np.abs(j).sum() == 0:
                continue
            for a, c in EDGES:
                ax.plot([j[a, 0], j[c, 0]], [j[a, 2], j[c, 2]],
                        [j[a, 1], j[c, 1]], color='tab:blue', lw=2)
            ax.scatter(j[:, 0], j[:, 2], j[:, 1], color='tab:red', s=8)
        ax.set_title('%s   frame %d/%d' % (ann['frame_dir'], t + 1, T), fontsize=9)

    anim = FuncAnimation(fig, draw, frames=T)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=15), dpi=80)
    print('Sample:', ann['frame_dir'], '| bodies drawn:', len(valid), '| frames:', T)
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
