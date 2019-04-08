#!/usr/bin/python3

import cv2
import os
import os.path as osp
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('files', type=str, nargs='+', help='')
    parser.add_argument('--outdir', '-o', type=str, default='images', help='')
    args = parser.parse_args()

    for video_path in args.files:
        print('Read video: %s' % video_path)
        images_dir = osp.join(args.outdir, ''.join(osp.basename(video_path).split('.')[:-1]))
        print('Output directory: %s' % images_dir)
        if not osp.isdir(images_dir):
            os.makedirs(images_dir)
        cap_video = cv2.VideoCapture(video_path)

        frame_id = 0
        while(cap_video.isOpened()):
            ret, image = cap_video.read()
            if ret:
                image = cv2.resize(image, (1920, 1080))
                dst_file_path = osp.join(images_dir, 'frame_%s.jpg' % frame_id)
                cv2.imwrite(dst_file_path, image)
            else:
                break
            if not frame_id % 1000:
                print(dst_file_path)
            frame_id += 1


if __name__ == '__main__':
    main()
