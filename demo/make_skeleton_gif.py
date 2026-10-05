import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import matplotlib
matplotlib.use('Agg')
 
from vision_gif import read_xyz, create_2D_animation, create_3D_animation
 
 
def main():
    parser = argparse.ArgumentParser(description='Render an NTU .skeleton file to animated GIFs')
    parser.add_argument('skeleton_file', help='path to the .skeleton file')
    parser.add_argument('--out-prefix', default='work_dirs/skeleton_demo',
                        help='output path prefix (no extension)')
    parser.add_argument('--mode', choices=['2d', '3d', 'both'], default='both')
    args = parser.parse_args()
 
    point = read_xyz(args.skeleton_file)
    print('Read Data Done!')
    num_frame = point.shape[1]
    print('Shape (coords x frames x joints x bodies):', point.shape)
 
    arms = [23, 11, 10, 9, 8, 20, 4, 5, 6, 7, 21]
    rightHand = [11, 24]
    leftHand = [7, 22]
    legs = [19, 18, 17, 16, 0, 12, 13, 14, 15]
    body = [3, 2, 20, 1, 0]
 
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
 
    if args.mode in ('2d', 'both'):
        p2 = f'{args.out_prefix}_2D.gif'
        create_2D_animation(num_frame, point, arms, rightHand, leftHand, legs, body, p2)
        print('Saved:', p2)
 
    if args.mode in ('3d', 'both'):
        p3 = f'{args.out_prefix}_3D.gif'
        create_3D_animation(num_frame, point, arms, rightHand, leftHand, legs, body, p3)
        print('Saved:', p3)
 
 
if __name__ == '__main__':
    main()
