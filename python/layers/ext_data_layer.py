import numpy as np
import cv2
import yaml
import caffe
from scipy.ndimage.filters import gaussian_filter
from collections import namedtuple


SampleDesc = namedtuple('SampleDesc', 'path label')


class SampleDataFromDisk:
    def __init__(self, folders_path):
        self._all_samples = {}
        self._ids_by_label = {}

        with open(folders_path) as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                arr = line.split()

                image_path = arr[0]
                label = int(arr[1])

                self._all_samples[i] = SampleDesc(path=image_path, label=label)
                self._ids_by_label[label] = self._ids_by_label.get(label, []) + [i]

        print('Number of classes: {}'.format(len(self._ids_by_label)))
        print('Number of training images: {}'.format(len(self._all_samples)))

    def get_image(self, sample_id):
        sample_desc = self._all_samples[sample_id]
        image = cv2.imread(sample_desc.path)

        return image, sample_desc.label

    def get_all_ids(self):
        return self._all_samples.keys()

    def get_ids_by_labels(self, label):
        return self._ids_by_label.get(label, [])

    def get_all_labels(self):
        return self._ids_by_label.keys()


class ExtDataLayer(caffe.Layer):
    @staticmethod
    def _image_to_blob(img, img_width, img_height, scales, subtract):
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        blob = img.astype(np.float32)

        if scales is not None:
            blob *= scales

        if subtract is not None:
            blob -= subtract

        blob = blob.transpose((2, 0, 1))

        return blob

    def _shuffle_data(self):
        if self.num_images_ == 1:
            self.data_ids_ = self.data_sampler_.get_all_ids()
            np.random.shuffle(self.data_ids_)
        else:
            all_labels = np.copy(self.data_sampler_.get_all_labels())
            np.random.shuffle(all_labels)

            self.data_ids_ = []
            for label in all_labels:
                label_ids = self.data_sampler_.get_ids_by_labels(label)
                if len(label_ids) <= 0:
                    continue

                self.data_ids_.append(np.random.choice(label_ids, self.num_images_, replace=True))

            self.data_ids_ = np.array(self.data_ids_).reshape([-1])

    def _augment(self, img, trg_height, trg_width):
        augmented_img = img

        if self.dither_:
            crop_aspect_ratio = np.random.uniform(self.aspect_ratio_limits_[0], self.aspect_ratio_limits_[1])
            crop_height = trg_height
            crop_width = int(float(crop_height) / crop_aspect_ratio)

            border_size = np.random.randint(1, self.max_border_size_)
            region_height = crop_height + 2 * border_size
            region_width = crop_width + 2 * border_size

            src_aspect_ratio = float(augmented_img.shape[0]) / float(augmented_img.shape[1])
            trg_aspect_ratio = float(region_height) / float(region_width)

            if src_aspect_ratio > trg_aspect_ratio:
                width = region_width
                height = int(float(width) * src_aspect_ratio)
            else:
                height = region_height
                width = int(float(height) / src_aspect_ratio)
            augmented_img = cv2.resize(augmented_img, (width, height))

            height_diff = augmented_img.shape[0] - crop_height
            width_diff = augmented_img.shape[1] - crop_width

            left_edge = np.random.randint(0, width_diff + 1)
            right_edge = left_edge + crop_width
            top_edge = np.random.randint(0, height_diff + 1)
            bottom_edge = top_edge + crop_height

            augmented_img = augmented_img[top_edge:bottom_edge, left_edge:right_edge]
            augmented_img = cv2.resize(augmented_img, (trg_width, trg_height))

        if self.blur_:
            if np.random.randint(0, 2) == 1:
                filter_size = np.random.uniform(low=self.sigma_limits_[0], high=self.sigma_limits_[1])

                augmented_img[:, :, 0] = gaussian_filter(augmented_img[:, :, 0], sigma=filter_size)
                augmented_img[:, :, 1] = gaussian_filter(augmented_img[:, :, 1], sigma=filter_size)
                augmented_img[:, :, 2] = gaussian_filter(augmented_img[:, :, 2], sigma=filter_size)

        if self.mirror_:
            if np.random.randint(0, 2) == 1:
                augmented_img = augmented_img[:, ::-1, :]

        if self.gamma_:
            rand = np.random.randint(0, 2)
            if rand == 1:
                u = np.random.uniform(-self.delta_, self.delta)
                gamma = np.log(0.5 + (2 ** (-0.5)) * u) / np.log(0.5 - (2 ** (-0.5)) * u)

                float_image = augmented_img.astype(np.float32) * (1. / 255.)
                augmented_img = (np.power(float_image, gamma) * 255.0).astype(np.int32)
                augmented_img[augmented_img > 255] = 255
                augmented_img[augmented_img < 0] = 0
                augmented_img = augmented_img.astype(np.uint8)

        if self.brightness_:
            rand = np.random.randint(0, 2)
            if rand == 1:
                if np.average(augmented_img) > self.min_pos_:
                    alpha = np.random.uniform(self.pos_alpha_[0], self.pos_alpha_[1])
                    beta = np.random.randint(self.pos_beta_[0], self.pos_beta_[1])
                else:
                    alpha = np.random.uniform(self.neg_alpha_[0], self.neg_alpha_[1])
                    beta = np.random.randint(self.neg_beta_[0], self.neg_beta_[1])

                augmented_img = (augmented_img.astype(np.float32) * alpha + beta).astype(np.int32)
                augmented_img[augmented_img > 255] = 255
                augmented_img[augmented_img < 0] = 0
                augmented_img = augmented_img.astype(np.uint8)

        if self.erase_:
            if np.random.randint(0, 2) == 1:
                width = augmented_img.shape[1]
                height = augmented_img.shape[0]

                num_erase_iter = np.random.randint(self.erase_num_[0], self.erase_num_[1])
                for _ in xrange(num_erase_iter):
                    erase_width = int(np.random.uniform(self.erase_size_[0], self.erase_size_[1]) * width)
                    erase_height = int(np.random.uniform(self.erase_size_[0], self.erase_size_[1]) * height)

                    left_edge = int(np.random.uniform(self.erase_border_[0], self.erase_border_[1]) * width)
                    top_edge = int(np.random.uniform(self.erase_border_[0], self.erase_border_[1]) * height)
                    right_edge = np.minimum(left_edge + erase_width, width)
                    bottom_edge = np.minimum(top_edge + erase_height, height)

                    if np.random.randint(0, 2) == 1:
                        fill_color = np.random.randint(0, 255, size=[bottom_edge - top_edge,
                                                                     right_edge - left_edge, 3], dtype=np.uint8)
                    else:
                        fill_color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                    augmented_img[top_edge:bottom_edge, left_edge:right_edge] = fill_color

        return augmented_img.astype(np.uint8)

    def _sample_next_batch(self):
        if self._index + self.batch_size_ > len(self.data_ids_):
            self._shuffle_data()
            self._index = 0

        sample_ids = self.data_ids_[self._index:(self._index + self.batch_size_)]
        self._index += self.batch_size_

        images_blob = []
        labels_blob = []

        for i in xrange(self.batch_size_):
            image, label = self.data_sampler_.get_image(sample_ids[i])

            augmented_image = self._augment(image, self.height_, self.width_)

            labels_blob.append(label)
            images_blob.append(self._image_to_blob(augmented_image,
                                                   self.width_, self.height_,
                                                   self.scales_, self.subtract_))

        return np.array(images_blob), np.array(labels_blob)

    def _set_data(self, data_sampler):
        self.data_sampler_ = data_sampler

        self._shuffle_data()

    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        assert 'num_ids' in layer_params
        assert 'num_images_per_id' in layer_params
        assert 'input_type' in layer_params
        assert 'height' in layer_params
        assert 'width' in layer_params

        self.num_ids_ = layer_params['num_ids']
        self.num_images_ = layer_params['num_images_per_id']
        self.batch_size_ = self.num_ids_ * self.num_images_
        self.height_ = layer_params['height']
        self.width_ = layer_params['width']

        assert self.num_ids_ > 0
        assert self.num_images_ > 0

        self.scales_ = layer_params['scales'] if 'scales' in layer_params else None
        self.subtract_ = layer_params['subtract'] if 'subtract' in layer_params else None

        self.blur_ = layer_params['blur'] if 'blur' in layer_params else False
        if self.blur_:
            self.sigma_limits_ = layer_params['sigma_limits'] if 'sigma_limits' in layer_params else [0.0, 0.5]
            assert 0.0 <= self.sigma_limits_[0] < self.sigma_limits_[1]

        self.gamma_ = layer_params['gamma'] if 'gamma' in layer_params else False
        if self.gamma_:
            self.delta_ = layer_params['delta'] if 'delta' in layer_params else 0.01
            assert 0.0 < self.delta_ < 1.0

        self.brightness_ = layer_params['brightness'] if 'brightness' in layer_params else False
        if self.brightness_:
            self.min_pos_ = layer_params['min_pos'] if 'min_pos' in layer_params else 100.0
            self.pos_alpha_ = layer_params['pos_alpha'] if 'pos_alpha' in layer_params else [0.2, 1.5]
            self.pos_beta_ = layer_params['pos_beta'] if 'pos_beta' in layer_params else [-100.0, 50.0]
            self.neg_alpha_ = layer_params['neg_alpha'] if 'neg_alpha' in layer_params else [0.9, 1.5]
            self.neg_beta_ = layer_params['neg_beta'] if 'neg_beta' in layer_params else [-20.0, 50.0]

        self.dither_ = layer_params['dither'] if 'dither' in layer_params else False
        if self.dither_:
            self.aspect_ratio_limits_ = layer_params['aspect_ratio_limits']\
                if 'aspect_ratio_limits' in layer_params else [1.8, 3.2]
            self.max_border_size_ = layer_params['max_border_size'] if 'max_border_size' in layer_params else 6
            assert 0.0 < self.aspect_ratio_limits_[0] < self.aspect_ratio_limits_[1]
            assert 1 < self.max_border_size_

        self.erase_ = layer_params['erase'] if 'erase' in layer_params else False
        if self.erase_:
            self.erase_num_ = layer_params['erase_num'] if 'erase_num' in layer_params else [1, 6]
            self.erase_size_ = layer_params['erase_size'] if 'erase_size' in layer_params else [0.2, 0.4]
            self.erase_border_ = layer_params['erase_border'] if 'erase_border' in layer_params else [0.1, 0.9]
            assert 0 < self.erase_num_[0] < self.erase_num_[1]
            assert 0.0 < self.erase_size_[0] < self.erase_size_[1] < 1.0
            assert 0.0 < self.erase_border_[0] < self.erase_border_[1] < 1.0

        self.mirror_ = layer_params['mirror'] if 'mirror' in layer_params else False

        if layer_params['input_type'] == 'lmdb':
            assert 'lmdb_path' in layer_params
            raise Exception('Not implemented source type: {}'.format(layer_params['input_type']))
        elif layer_params['input_type'] == 'list':
            assert 'file_path' in layer_params
            data_sampler = SampleDataFromDisk(layer_params['file_path'])
        else:
            raise Exception('Unknown input format: {}'.format(layer_params['input_type']))
        self._set_data(data_sampler)

    def _init_states(self):
        self._index = 0

    def setup(self, bottom, top):
        self._load_params(self.param_str)
        self._init_states()

    def forward(self, bottom, top):
        images_blob, labels_blob = self._sample_next_batch()

        top[0].data[...] = images_blob
        top[1].data[...] = labels_blob

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size_, 3, self.height_, self.width_)
        top[1].reshape(self.batch_size_)
