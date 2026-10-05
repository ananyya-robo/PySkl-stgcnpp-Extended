## 导入第三方库
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import sys

## 读取关节数据
def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []

        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }

                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []

                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)

                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence

## 读取关节的x，y，z三个坐标
def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)

    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data

## 2D动画并保存为动图
def create_2D_animation(num_frame, point, arms, rightHand, leftHand, legs, body, gif_path):
    fig, ax = plt.subplots()

    # 求坐标最大值
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :])
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])

    def update(i):
        ax.clear()
        # 画出两个body所有关节
        ax.scatter(point[0, i, :, :], point[1, i, :, :], c='red', s=40.0)
        # 连接第一个body的关节，形成骨骼
        ax.plot(point[0, i, arms, 0], point[1, i, arms, 0], c='green', lw=2.0)
        ax.plot(point[0, i, rightHand, 0], point[1, i, rightHand, 0], c='green', lw=2.0)
        ax.plot(point[0, i, leftHand, 0], point[1, i, leftHand, 0], c='green', lw=2.0)
        ax.plot(point[0, i, legs, 0], point[1, i, legs, 0], c='green', lw=2.0)
        ax.plot(point[0, i, body, 0], point[1, i, body, 0], c='green', lw=2.0)

        # 连接第二个body的关节，形成骨骼
        ax.plot(point[0, i, arms, 1], point[1, i, arms, 1], c='green', lw=2.0)
        ax.plot(point[0, i, rightHand, 1], point[1, i, rightHand, 1], c='green', lw=2.0)
        ax.plot(point[0, i, leftHand, 1], point[1, i, leftHand, 1], c='green', lw=2.0)
        ax.plot(point[0, i, legs, 1], point[1, i, legs, 1], c='green', lw=2.0)
        ax.plot(point[0, i, body, 1], point[1, i, body, 1], c='green', lw=2.0)

        ax.text(xmax, ymax + 0.2, 'frame: {}/{}'.format(i, num_frame - 1))
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_ylim(ymin - 0.3, ymax + 0.3)

    ani = FuncAnimation(fig, update, frames=num_frame, repeat=False)
    ani.save(gif_path, writer=PillowWriter(fps=10))

    plt.close()

## 3D动画并保存为动图
def create_3D_animation(num_frame, point, arms, rightHand, leftHand, legs, body, gif_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 求坐标最大值
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :])
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])

    Expan_Multiple = 1.4  # 坐标扩大倍数，绘图较美观

    def update(i):
        ax.clear()
        ax.view_init(120, -90)

        # 画出两个body所有关节
        ax.scatter(point[0, i, :, :] * Expan_Multiple, point[1, i, :, :] * Expan_Multiple, point[2, i, :, :],
                   c='red', s=40.0)

        # 连接第一个body的关节，形成骨骼
        ax.plot(point[0, i, arms, 0] * Expan_Multiple, point[1, i, arms, 0] * Expan_Multiple, point[2, i, arms, 0],
                c='green', lw=2.0)
        ax.plot(point[0, i, rightHand, 0] * Expan_Multiple, point[1, i, rightHand, 0] * Expan_Multiple,
                point[2, i, rightHand, 0], c='green', lw=2.0)
        ax.plot(point[0, i, leftHand, 0] * Expan_Multiple, point[1, i, leftHand, 0] * Expan_Multiple,
                point[2, i, leftHand, 0], c='green', lw=2.0)
        ax.plot(point[0, i, legs, 0] * Expan_Multiple, point[1, i, legs, 0] * Expan_Multiple, point[2, i, legs, 0],
                c='green', lw=2.0)
        ax.plot(point[0, i, body, 0] * Expan_Multiple, point[1, i, body, 0] * Expan_Multiple, point[2, i, body, 0],
                c='green', lw=2.0)

        # 连接第二个body的关节，形成骨骼
        ax.plot(point[0, i, arms, 1] * Expan_Multiple, point[1, i, arms, 1] * Expan_Multiple, point[2, i, arms, 1],
                c='green', lw=2.0)
        ax.plot(point[0, i, rightHand, 1] * Expan_Multiple, point[1, i, rightHand, 1] * Expan_Multiple,
                point[2, i, rightHand, 1], c='green', lw=2.0)
        ax.plot(point[0, i, leftHand, 1] * Expan_Multiple, point[1, i, leftHand, 1] * Expan_Multiple,
                point[2, i, leftHand, 1], c='green', lw=2.0)
        ax.plot(point[0, i, legs, 1] * Expan_Multiple, point[1, i, legs, 1] * Expan_Multiple, point[2, i, legs, 1],
                c='green', lw=2.0)
        ax.plot(point[0, i, body, 1] * Expan_Multiple, point[1, i, body, 1] * Expan_Multiple, point[2, i, body, 1],
                c='green', lw=2.0)

        ax.text(xmax - 0.3, ymax + 1.1, zmax + 0.3, 'frame: {}/{}'.format(i, num_frame - 1))
        ax.set_xlim(xmin - 0.5, xmax + 0.5)
        ax.set_ylim(ymin - 0.3, ymax + 0.3)
        ax.set_zlim(zmin - 0.3, zmax + 0.3)

    ani = FuncAnimation(fig, update, frames=num_frame, repeat=False)
    ani.save(gif_path, writer=PillowWriter(fps=10))

    plt.close()

## main函数
def main():
    sys.path.extend(['../'])  # 扩展路径
    data_path = '/Users/oumurasakikaya/Desktop/individual project/pyskl/demo/S001C003P008R002A045.skeleton'  # 拍手skeleton文件名
    point = read_xyz(data_path)  # 读取 x,y,z三个坐标
    print('Read Data Done!')  # 数据读取完毕

    num_frame = point.shape[1]  # 帧数
    print(point.shape)  # 坐标数(3) × 帧数 × 关节数(25) × max_body(2)

    # 相邻关节标号
    arms = [23, 11, 10, 9, 8, 20, 4, 5, 6, 7, 21]
    rightHand = [11, 24]
    leftHand = [7, 22]
    legs = [19, 18, 17, 16, 0, 12, 13, 14, 15]
    body = [3, 2, 20, 1, 0]

    # 生成2D动图
    gif_path_2d = "/Users/oumurasakikaya/Desktop/individual project/skeleton_animation_2D_45.gif"
    create_2D_animation(num_frame, point, arms, rightHand, leftHand, legs, body, gif_path_2d)

    # 生成3D动图
    gif_path_3d = "/Users/oumurasakikaya/Desktop/individual project/skeleton_animation_3D_45.gif"
    create_3D_animation(num_frame, point, arms, rightHand, leftHand, legs, body, gif_path_3d)

if __name__ == "__main__":
    main()
