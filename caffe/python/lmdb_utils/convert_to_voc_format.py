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

from os import path, makedirs
import argparse
import glog as log
from tqdm import tqdm
from PIL import Image as pil_image
import json

MODES = ['L', 'RGB']

#pylint: disable=unused-variable, invalid-name
def save_image_annotation(ann, image_id, file_path, labels_of_interest=None):
    """ Save image annotation
    """
    import xml.etree.cElementTree as ET

    root = ET.Element('annotation')

    ET.SubElement(root, 'filename').text = image_id
    try:
        img = pil_image.open(ann['image'])
        width, height = img.size
        mode = img.mode
        assert mode in MODES
        depth = len(mode)

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)
        ET.SubElement(size, 'depth').text = str(depth)
    except Exception as ex:
        tqdm.write('invalid image {}'.format(ann['image']))
        return 0, 0
    ET.SubElement(root, 'segmented').text = '0'
    for obj in ann['objects']:
        obj_node = ET.SubElement(root, 'object')
        label = obj.get('label', 'ignored')
        if label == 'ignored':
            continue
        if labels_of_interest is not None and label not in labels_of_interest:
            continue
        ET.SubElement(obj_node, 'name').text = label
        ET.SubElement(obj_node, 'pose').text = 'Unspecified'
        ET.SubElement(obj_node, 'truncated').text = '0'
        ET.SubElement(obj_node, 'difficult').text = '1' if label == 'ignored' else '0'
        bndbox = ET.SubElement(obj_node, 'bndbox')
        bbox = list(map(int, obj['bbox']))
        ET.SubElement(bndbox, 'xmin').text = str(bbox[0])
        ET.SubElement(bndbox, 'ymin').text = str(bbox[1])
        ET.SubElement(bndbox, 'xmax').text = str(bbox[2])
        ET.SubElement(bndbox, 'ymax').text = str(bbox[3])
    tree = ET.ElementTree(root)
    tree.write(file_path)
    return width, height


def main():
    """ Convert original annotation to Pascal VOC format compatible with Caffe SSD tools.
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('annotation_file_path')
    parser.add_argument('out_dir_path')
    parser.add_argument('out_list_file_path')
    parser.add_argument('--name_size_file_path', default='')
    parser.add_argument('--labels', default=None, nargs='+')
    args = parser.parse_args()

    ann_list = []

    if path.exists(args.out_dir_path) and not path.isdir(args.out_dir_path):
        log.info('{} is not a directory'.format(args.out_dir_path))
        exit()

    if not path.exists(path.join(args.out_dir_path, 'voc_annotation')):
        makedirs(path.join(args.out_dir_path, 'voc_annotation'))

    log.info('loading annotation...')
    annotation = json.load(open(args.annotation_file_path, 'r'))
    image_sizes = []
    log.info('loading annotation...[done]')
    log.info('processing each image...')
    for image_idx, image_annotation in tqdm(enumerate(annotation), total=len(annotation)):
        image_id = '{:07}'.format(image_idx)
        ann_file_path = path.join(args.out_dir_path, 'voc_annotation', image_id + '.xml')
        ann_list.append((image_annotation['image'], ann_file_path))
        width, height = save_image_annotation(image_annotation, image_id, ann_file_path,
                                              labels_of_interest=args.labels)
        image_sizes.append((width, height))
    log.info('processing each image...[done]')

    log.info('saving annotation list...')
    open(args.out_list_file_path, 'wt').write('\n'.join([' '.join(x) for x in ann_list]))
    log.info('saving annotation list...[done]')

    log.info('obtaining size of each image...')
    if args.name_size_file_path:
        with open(args.name_size_file_path, 'wt') as f:
            for (image_id, ann_path), (width, height) in tqdm(zip(ann_list, image_sizes)):
                f.write('{} {} {}\n'.format(image_id, height, width))
    log.info('obtaining size of each image...[done]')

if __name__ == "__main__":
    main()
