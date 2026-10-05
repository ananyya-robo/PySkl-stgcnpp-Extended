import argparse
import pickle
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
 
from pyskl.models import build_model
from pyskl.datasets.pipelines import Compose
 
 
def load_label_map(path, num_classes):
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    return names[:num_classes]
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset_pkl', default='data/nturgbd/ntu60_3danno.pkl')
    parser.add_argument('--split', default='xsub_val')
    parser.add_argument('--label_map', default='tools/data/label_map/nturgbd_120.txt')
    parser.add_argument('--target_class', type=int, default=42,
                        help='0-indexed class; 42 = A43 falling')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='which matching sample to use (0 = first)')
    parser.add_argument('--topk', type=int, default=3)
    args = parser.parse_args()
 
    cfg = Config.fromfile(args.config)
 
    with open(args.dataset_pkl, 'rb') as f:
        data = pickle.load(f)
 
    val_names = set(data['split'][args.split])
    matches = [a for a in data['annotations']
               if a['frame_dir'] in val_names and a['label'] == args.target_class]
 
    if not matches:
        print(f'No samples found for class {args.target_class} in {args.split}')
        return
 
    print(f'Found {len(matches)} samples of this class in {args.split}')
    ann = matches[args.sample_idx]
 
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda().eval()
 
    pipeline = Compose(cfg.data.test.pipeline)
    sample = pipeline(dict(**ann, modality='Pose', test_mode=True, start_index=0))
 
    keypoint = sample['keypoint']
    if not torch.is_tensor(keypoint):
        keypoint = torch.from_numpy(np.asarray(keypoint))
    keypoint = keypoint.float().unsqueeze(0).cuda()
 
    with torch.no_grad():
        scores = model(keypoint, return_loss=False)
 
    scores = np.asarray(scores).squeeze()
    if scores.ndim > 1:
        scores = scores.mean(axis=0)
 
    names = load_label_map(args.label_map, len(scores))
    pred = int(np.argmax(scores))
    order = np.argsort(scores)[::-1][:args.topk]
 
    print()
    print('=' * 52)
    print(f'Sample:           {ann["frame_dir"]}')
    print(f'True class:       A{args.target_class + 1} - {names[args.target_class]}')
    print(f'Predicted class:  A{pred + 1} - {names[pred]}')
    print(f'Confidence:       {scores[pred]:.4f}')
    print(f'Correct:          {"YES" if pred == args.target_class else "NO"}')
    print()
    print(f'Top-{args.topk} predictions:')
    for rank, i in enumerate(order, 1):
        print(f'  {rank}. {names[i]:<45s} {scores[i]:.4f}')
    print('=' * 52)
 
 
if __name__ == '__main__':
    main()
