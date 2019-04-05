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

import json
import cv2

class ImageGrabberInterface(object):
    def __init__(self, *args, **kwargs):
        super(ImageGrabberInterface, self).__init__()

    def __enter__(self):
        return self

    def close(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_image(self, image_path):
        raise NotImplementedError

    def get_image_size(self, image_path):
        raise NotImplementedError

    def copy_image(self, src_image_path, dst_image_path, default_extension='.jpg'):
        raise NotImplementedError

class ImageGrabber(ImageGrabberInterface):

    def __init__(self, *args, **kwargs):
        super(ImageGrabber, self).__init__(*args, **kwargs)

    def close(self):
        return

    def get_image(self, image_path):
        return cv2.imread(image_path)


class IterableImageGrabberInterface(object):

    @staticmethod
    def create_iterable_image_grabber(grabber_type, *args, **kwargs):
        if grabber_type == 'annotation':
            return AnnotationImageGrabber(*args, **kwargs)
        else:
            raise ValueError('Unknown iterable image grabber type {}'.format(grabber_type))

    def __init__(self, *args, **kwargs):
        super(IterableImageGrabberInterface, self).__init__()
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    next = __next__

    def __getitem__(self, position):
        raise NotImplementedError

class AnnotationImageGrabber(IterableImageGrabberInterface):
    def __init__(self, annotation_file_path, *args, **kwargs):
        super(AnnotationImageGrabber, self).__init__(*args, **kwargs)
        self.annotation = json.load(open(annotation_file_path, 'r'))
        self.counter = 0
        self.image_grabber = ImageGrabber()

    def __len__(self):
        return len(self.annotation)

    def __next__(self):
        if self.counter < len(self):
            image_path = self.annotation[self.counter]['image']
            image = self.image_grabber.get_image(image_path)
            self.counter += 1
            return image, image_path
        else:
            raise StopIteration

    def next(self):
        if self.counter < len(self):
            image_path = self.annotation[self.counter]['image']
            image = self.image_grabber.get_image(image_path)
            self.counter += 1
            return image, image_path
        else:
            raise StopIteration

    def close(self):
        self.image_grabber.close()
