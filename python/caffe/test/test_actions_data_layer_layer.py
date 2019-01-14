# pylint: skip-file

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

import shutil
import tempfile
import unittest

from os import remove
from os.path import join

import caffe
from PIL.Image import fromarray as image_from_array
import numpy as np


def python_net_file(tasks_path, batch_size, height, width):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        proto_begin = """name: 'actions_data_layer_test_net'
                         layer { type: 'Python' name: 'actions_data' top: 'actions_data' top: 'actions_labels'
                                 python_param { module: 'caffe.custom_layers.actions_data_layer'
                                                layer: 'ActionsDataLayer'
                                                param_str: '{ """
        template_str = '"tasks": "{}", "batch": {}, "height": {}, "width": {}, "valid_action_ids": [0, 1, 2, 3], ' \
                       '"ignore_class_id": 3, "num_data_fillers": 1, "data_queue_size": 6, "single_iter": True,' \
                       '"expand_weight": 0.0, "crop_weight": 0.0, "free_weight": 1.0' \
            .format(tasks_path, batch_size, height, width)
        proto_end = """ }' } }"""

        f.write(proto_begin + template_str + proto_end)

        return f.name


def create_directory_with_images(height, width, num_images):
    tmp_directory = tempfile.mkdtemp()
    for i in xrange(num_images):
        image_array = np.random.randint(low=0, high=256, size=(height, width, 3), dtype=np.uint8)

        file_name = join(tmp_directory, 'frame_{:06}.png'.format(i))
        image_from_array(image_array).save(file_name)

    return tmp_directory


def create_annotation_file(num_frames, num_tracks):
    actions_str = ['sitting', 'standing', 'raising_hand']

    doc_begin = '<annotations count="{}">'
    doc_end = '</annotations>'

    track_begin = '<track id="{}" label="person">'
    track_end = '</track>'

    bbox_template = '<box frame="{}" xtl="1.0" ytl="1.0" xbr="0.0" ybr="0.0" outside="0" occluded="0" keyframe="1">' \
                    '<attribute name="action">{}</attribute></box>'

    out_str = doc_begin.format(num_tracks)
    for track_id in xrange(num_tracks):
        out_str += track_begin.format(track_id)
        for frame_id in xrange(num_frames):
            out_str += bbox_template.format(frame_id, actions_str[np.random.randint(len(actions_str))])
        out_str += track_end
    out_str += doc_end

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.xml') as f:
        f.write(out_str)

        return f.name


def create_tasks_file(annotation_path, images_directory, height, width):
    template_str = '{} {},{} {}'.format(annotation_path, height, width, images_directory)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as f:
        f.write(template_str)

        return f.name


@unittest.skipIf('Python' not in caffe.layer_type_list(), 'Caffe built without Python layer support')
class TestActionsDataLayer(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward(self):
        num_images = 6
        num_tracks = 7
        batch_size = 3
        input_size = (20, 45)
        output_size = (11, 23)

        images_directory_path =\
            create_directory_with_images(height=input_size[0], width=input_size[1], num_images=num_images)
        annotation_path = create_annotation_file(num_images, num_tracks)
        tasks_path = create_tasks_file(annotation_path, images_directory_path, input_size[0], input_size[1])

        net_file = python_net_file(tasks_path, batch_size, output_size[0], output_size[1])
        net = caffe.Net(net_file, caffe.TRAIN)
        remove(net_file)

        net_outputs = net.forward()
        out_data = net_outputs['actions_data']
        out_labels = net_outputs['actions_labels']

        shutil.rmtree(images_directory_path)
        remove(annotation_path)
        remove(tasks_path)

        self.assertTupleEqual(out_data.shape, (batch_size, 3, output_size[0], output_size[1]))
        self.assertTupleEqual(out_labels.shape, (1, 1, batch_size * num_tracks, 8))
