"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os, sys
import h5py
import math
import argparse
import numpy as np
import cv2
from random import randint, shuffle

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('annotation_file_path')
    parser.add_argument('output_file_name')
    parser.add_argument('--width', default=62)
    parser.add_argument('--height', default=62)
    return parser.parse_args()

def main():
    args = parse_args()

    file_cont = open(args.annotation_file_path, 'r').readlines()
    data_dir = os.path.dirname(args.annotation_file_path)
    print("dir name = ", data_dir)
    modelWidth = int(args.width)
    modelHeight = int(args.height)
    output_name = args.output_file_name

    samples_count = len(file_cont)
    spl = file_cont[0].split()
    labels_dim = len(spl) - 1

    data_file_path = os.path.join(data_dir, spl[0])
    print('data_file_path = ', data_file_path)
    im = cv2.imread(data_file_path)
    im = cv2.resize(im, (modelWidth, modelHeight), interpolation = cv2.INTER_CUBIC)

    data = 1 + np.arange(modelWidth * modelHeight * 3 * samples_count, dtype=np.float32 )[:, np.newaxis]
    data = data.reshape(samples_count, 3, modelWidth, modelHeight)

    label = 1 + np.arange(labels_dim * samples_count)[:, np.newaxis]
    label = label.reshape(samples_count, labels_dim)
    label = label.astype('float32')

    idx = 0

    last_spl = spl
    last_im = im
    for l in file_cont[0:]:
        print(idx, 'of', samples_count)

        spl = l.split()

        im = np.array(modelWidth*modelHeight)
        size = 1
        try:
            im1 = cv2.imread(os.path.join(data_dir, spl[0]))
            print('im1 path = ', os.path.join(data_dir,spl[0]))
            im = cv2.resize(im1, (modelWidth, modelHeight), interpolation = cv2.INTER_CUBIC)
            im = im.astype(np.float32)
            im = im / 255.0
        except:
            data[idx] = last_im

            last_spl[2] = float(last_spl[2])
            for i in range(labels_dim):
                label[idx][i] = float(last_spl[i+1])
            idx += 1

            continue

        im = im.transpose((2, 0, 1))
        data[idx] = im
        last_im = im

        spl[2] = float(spl[2])
        for i in range(labels_dim):
            label[idx][i] = float(spl[i+1])
            print( label[idx][i])

        last_spl = spl

        idx += 1

    with h5py.File(data_dir + '/' + output_name + '.h5', 'w') as f:
                f['data'] = data
                f['label'] = label
    with open(data_dir + '/' + output_name + '_list.txt', 'a') as f:
        f.write(data_dir +  '/' + output_name + '.h5\n')


if __name__ == "__main__":
    main()
