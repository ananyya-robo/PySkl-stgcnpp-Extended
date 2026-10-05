import argparse
import pickle
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def expected_calibration_error(confidences, correct, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []
    for lo, hi in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(correct[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
    return ece, bin_accs, bin_confs, bin_counts, bin_lowers


def brier_score(scores, labels, num_classes):
    one_hot = np.eye(num_classes)[labels]
    return np.mean(np.sum((scores - one_hot) ** 2, axis=1))


def plot_reliability(bin_lowers, bin_accs, n_bins, title, out_path):
    width = 1.0 / n_bins
    centers = np.array(bin_lowers) + width / 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(centers, bin_accs, width=width, edgecolor='black', color='steelblue', label='Accuracy')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_subset(scores, labels, mask, num_classes, n_bins, label_str):
    s = scores[mask]
    y = labels[mask]
    if len(y) == 0:
        print(f"{label_str}: no samples")
        return None
    preds = np.argmax(s, axis=1)
    confidences = np.max(s, axis=1)
    correct = (preds == y).astype(float)
    top1_acc = np.mean(correct)
    ece, bin_accs, bin_confs, bin_counts, bin_lowers = expected_calibration_error(confidences, correct, n_bins)
    brier = brier_score(s, y, num_classes)
    mean_conf = np.mean(confidences)
    print(f"{label_str} (n={len(y)}):")
    print(f"  top1_acc: {top1_acc:.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier score: {brier:.4f}")
    print(f"  Mean confidence: {mean_conf:.4f}")
    return {
        'top1_acc': top1_acc, 'ece': ece, 'brier': brier, 'mean_conf': mean_conf,
        'bin_accs': bin_accs, 'bin_lowers': bin_lowers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_pkl', required=True)
    parser.add_argument('--dataset_pkl', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--n_bins', type=int, default=15)
    args = parser.parse_args()

    with open(args.result_pkl, 'rb') as f:
        scores = np.array(pickle.load(f))

    with open(args.dataset_pkl, 'rb') as f:
        data = pickle.load(f)

    test_names = data['split'][args.split]
    ann_by_name = {a['frame_dir']: a for a in data['annotations']}
    labels = np.array([ann_by_name[name]['label'] for name in test_names])

    num_classes = scores.shape[1]
    health_classes = list(range(40, 49))  # A41-A49, 0-indexed
    health_mask = np.isin(labels, health_classes)
    non_health_mask = ~health_mask

    print(f"=== {args.name} ===")
    all_res = evaluate_subset(scores, labels, np.ones(len(labels), dtype=bool), num_classes, args.n_bins, 'All classes')
    health_res = evaluate_subset(scores, labels, health_mask, num_classes, args.n_bins, 'Health classes (A41-A49)')
    non_health_res = evaluate_subset(scores, labels, non_health_mask, num_classes, args.n_bins, 'Non-health classes')

    if all_res:
        plot_reliability(all_res['bin_lowers'], all_res['bin_accs'], args.n_bins,
                         f'{args.name} - All classes', f'work_dirs/reliability_{args.name}_all.png')
    if health_res:
        plot_reliability(health_res['bin_lowers'], health_res['bin_accs'], args.n_bins,
                         f'{args.name} - Health classes (A41-A49)', f'work_dirs/reliability_{args.name}_health.png')

    print(f"Saved: work_dirs/reliability_{args.name}_all.png, work_dirs/reliability_{args.name}_health.png")


if __name__ == '__main__':
    main()