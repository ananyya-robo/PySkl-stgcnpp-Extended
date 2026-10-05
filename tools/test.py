# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv import digit_version as dv
from mmcv import load
from mmcv.cnn import fuse_conv_bn
from mmcv.engine import multi_gpu_test
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import sys

print(sys.path)
from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from pyskl.utils import cache_checkpoint, mc_off, mc_on, test_port

from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def calculate_metrics(y_true, y_pred, y_prob):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr')

    # 计算每个类的TPR和FPR
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm.diagonal() / cm.sum(axis=1)
    fpr = (cm.sum(axis=0) - cm.diagonal()) / cm.sum(axis=0)

    return precision, recall, auc_score, tpr.mean(), fpr.mean()


def plot_roc_curve(y_true, y_prob, n_classes, name='roc'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制每个类的ROC曲线
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'work_dirs/roc_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='pyskl test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('-C', '--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['top_k_accuracy', 'mean_class_accuracy'],
        help='evaluation metrics, which depends on the dataset, e.g.,'
             ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple workers')
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        '--compile',
        action='store_true',
        help='whether to compile the model before training / testing (only available in pytorch 2.0)')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def inference_pytorch(args, cfg, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg', dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    model = build_model(cfg.model)
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)

    if args.checkpoint is None:
        work_dir = cfg.work_dir
        args.checkpoint = osp.join(work_dir, 'latest.pth')
        assert osp.exists(args.checkpoint)

    args.checkpoint = cache_checkpoint(args.checkpoint)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    print("Outputs structure:", outputs[:5])  # 打印前5个输出的结构

    # 获取真实标签
    y_true = [data['label'] for data in data_loader.dataset]
    y_pred = [output.argmax() for output in outputs]
    y_prob = np.array(outputs)  # 假设 outputs 本身就是预测的概率分布

    precision, recall, auc_score, tpr, fpr = calculate_metrics(y_true, y_pred, y_prob)

    print(f'Precision: {precision:.04f}')
    print(f'Recall: {recall:.04f}')
    print(f'AUC: {auc_score:.04f}')
    print(f'TPR: {tpr:.04f}')
    print(f'FPR: {fpr:.04f}')

    cfg_name = osp.basename(osp.dirname(args.config))
    stream_name = osp.splitext(osp.basename(args.config))[0]
    plot_roc_curve(np.array(y_true), y_prob, y_prob.shape[1], name=f'{stream_name}_{cfg_name}')

    return outputs


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    out = osp.join(cfg.work_dir, 'result.pkl') if args.out is None else args.out

    eval_cfg = cfg.get('evaluation', {})
    keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
    for key in keys:
        eval_cfg.pop(key, None)
    if args.eval:
        eval_cfg['metrics'] = args.eval

    mmcv.mkdir_or_exist(osp.dirname(out))
    _, suffix = osp.splitext(out)
    assert suffix[1:] in file_handlers, ('The format of the output file should be json, pickle或yaml')

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        shuffle=False)
    dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    default_mc_cfg = ('localhost', 22077)
    memcached = cfg.get('memcached', False)

    if rank == 0 and memcached:
        mc_cfg = cfg.get('mc_cfg', default_mc_cfg)
        assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
        if not test_port(mc_cfg[0], mc_cfg[1]):
            mc_on(port=mc_cfg[1], launcher=args.launcher)
        retry = 3
        while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached. '

    dist.barrier()
    outputs = inference_pytorch(args, cfg, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        print(f'\nwriting results to {out}')
        dataset.dump_results(outputs, out=out)
        if eval_cfg:
            eval_res = dataset.evaluate(outputs, **eval_cfg)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')

    dist.barrier()
    if rank == 0 and memcached:
        mc_off()


if __name__ == '__main__':
    main()
