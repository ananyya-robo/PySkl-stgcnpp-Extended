import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

with open('work_dirs/j_result_xview.pkl', 'rb') as f:
    j_scores = np.array(pickle.load(f))

with open('work_dirs/b_result_xview.pkl', 'rb') as f:
    b_scores = np.array(pickle.load(f))

with open('data/nturgbd/ntu60_3danno.pkl', 'rb') as f:
    data = pickle.load(f)

test_names = data['split']['xview_val']
ann_by_name = {a['frame_dir']: a for a in data['annotations']}
labels = np.array([ann_by_name[name]['label'] for name in test_names])

fused_scores = (j_scores + b_scores) / 2

# Save the fused scores so calibration.py can read them
with open('work_dirs/2s_result_xview.pkl', 'wb') as f:
    pickle.dump(fused_scores, f)

num_classes = fused_scores.shape[1]
top1_pred = np.argmax(fused_scores, axis=1)
top1_acc = np.mean(top1_pred == labels)
top5_pred = np.argsort(fused_scores, axis=1)[:, -5:]
top5_acc = np.mean([labels[i] in top5_pred[i] for i in range(len(labels))])

precision = precision_score(labels, top1_pred, average='macro', zero_division=0)
recall = recall_score(labels, top1_pred, average='macro', zero_division=0)

y_true_bin = label_binarize(labels, classes=list(range(num_classes)))
auc = roc_auc_score(y_true_bin, fused_scores, average='macro', multi_class='ovr')

# Macro TPR / FPR from confusion matrix
tpr_list, fpr_list = [], []
for c in range(num_classes):
    tp = np.sum((top1_pred == c) & (labels == c))
    fn = np.sum((top1_pred != c) & (labels == c))
    fp = np.sum((top1_pred == c) & (labels != c))
    tn = np.sum((top1_pred != c) & (labels != c))
    tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
tpr = np.mean(tpr_list)
fpr = np.mean(fpr_list)

print(f"2s fused top1_acc: {top1_acc:.4f}")
print(f"2s fused top5_acc: {top5_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC: {auc:.4f}")
print(f"TPR: {tpr:.4f}")
print(f"FPR: {fpr:.4f}")
