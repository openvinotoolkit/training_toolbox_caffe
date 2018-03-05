import numpy as np
import cv2
import lmdb
import yaml
import caffe
from scipy.ndimage.filters import gaussian_filter
from random import shuffle, choice


class SampleDataFromLmdb(object):
    def __init__(self, lmdb_path):
        self._sample_table = {}

        self._cursor = lmdb.open(lmdb_path, readonly=True).begin().cursor()
        self._datum = caffe.proto.caffe_pb2.Datum()

        count = 0
        for (index, (key, value)) in enumerate(self._cursor):
            self._datum.ParseFromString(value)
            label = self._datum.label

            if label not in self._sample_table.keys():
                self._sample_table[label] = []

            self._sample_table[label].append([key, 0])
            self._sample_table[label].append([key, 1])

            count += 1

        print 'Number of classes:', len(self._sample_table)
        print 'Number of training images:', count

    def get_image(self, key):
        value = self._cursor.get(key)
        self._datum.ParseFromString(value)
        label = self._datum.label

        img = caffe.io.datum_to_array(self._datum)

        channel_swap = (1, 2, 0)
        img = img.transpose(channel_swap)

        return img, label

    def get_ids(self):
        return self._sample_table.keys()

    def get_data(self, key):
        return self._sample_table.get(key, [])


class SampleDataFromDisk(object):
    def __init__(self, folders_path):
        self._sample_table = {}
        self._sample_label = {}

        count = 0
        with open(folders_path) as f:
            for line in f.readlines():
                line = line.strip()
                arr = line.split()

                p = arr[0]
                label = int(arr[1])

                if label not in self._sample_table.keys():
                    self._sample_table[label] = []

                self._sample_table[label].append([p, 0])
                self._sample_table[label].append([p, 1])

                self._sample_label[p] = label

                count += 1

        print 'Number of classes:', len(self._sample_table)
        print 'Number of training images:', count, len(self._sample_label)

    def get_image(self, path):
        img = cv2.imread(path, 1)
        label = self._sample_label[path]

        return img, label

    def get_ids(self):
        return self._sample_table.keys()

    def get_data(self, key):
        return self._sample_table.get(key, [])


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
        all_ids = self.data_sampler_.get_ids()
        shuffle(all_ids)

        self.data_ = []
        for object_id in all_ids:
            object_paths = self.data_sampler_.get_data(object_id)
            if len(object_paths) <= 0:
                continue

            for _ in xrange(self.num_images_):
                self.data_.append(choice(object_paths))

    def _augment(self, img):
        augmented_img = img

        if self.dither_:
            width = float(augmented_img.shape[1])
            height = float(augmented_img.shape[0])

            left_edge = int(width * np.random.uniform(0.0, self.max_factor_left_))
            right_edge = int(width * (1.0 - np.random.uniform(0.0, self.max_factor_right_)))
            top_edge = int(height * np.random.uniform(0.0, self.max_factor_top_))
            bottom_edge = int(height * (1.0 - np.random.uniform(0.0, self.max_factor_bottom_)))

            crop = augmented_img[top_edge:bottom_edge, left_edge:right_edge]
            augmented_img = cv2.resize(crop, (width, height))

        if self.blur_:
            if np.random.randint(0, 2) == 1:
                filter_size = np.random.uniform(low=self.sigma_limits_[0], high=self.sigma_limits_[1])

                augmented_img[:, :, 0] = gaussian_filter(augmented_img[:, :, 0], sigma=filter_size)
                augmented_img[:, :, 1] = gaussian_filter(augmented_img[:, :, 1], sigma=filter_size)
                augmented_img[:, :, 2] = gaussian_filter(augmented_img[:, :, 2], sigma=filter_size)

        if self.mirror_:
            if np.random.randint(0, 2) == 1:
                augmented_img = augmented_img[:, ::-1, :]

        if self.change_brightness_:
            rand = np.random.randint(0, 2)
            if rand == 1:
                if np.average(augmented_img) > self.min_pos_:
                    alpha = np.random.uniform(self.pos_alpha_[0], self.pos_alpha_[1])
                    beta = np.random.randint(self.pos_beta_[0], self.pos_beta_[1])
                else:
                    alpha = np.random.uniform(self.neg_alpha_[0], self.neg_alpha_[1])
                    beta = np.random.randint(self.neg_beta_[0], self.neg_beta_[1])

                changed_brightness = augmented_img * alpha + beta

                augmented_img = np.where(changed_brightness < 255,
                                         changed_brightness,
                                         np.full_like(augmented_img, 255, dtype=np.uint8))
                augmented_img = np.where(augmented_img >= 0,
                                         augmented_img,
                                         np.full_like(augmented_img, 0, dtype=np.uint8))

        if self.erase_:
            if np.random.randint(0, 2) == 1:
                width = augmented_img.shape[1]
                height = augmented_img.shape[0]

                max_erase_width = np.minimum(int(self.erase_max_size_ * width), width - 1)
                max_erase_height = np.minimum(int(self.erase_max_size_ * height), height - 1)

                left_edge = int(np.random.uniform(self.erase_border_[0], self.erase_border_[1]) * width)
                top_edge = int(np.random.uniform(self.erase_border_[0], self.erase_border_[1]) * height)
                right_edge = np.random.randint(left_edge, max_erase_width)
                bottom_edge = np.random.randint(top_edge, max_erase_height)

                fill_color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                augmented_img[top_edge:bottom_edge, left_edge:right_edge] = fill_color

        return augmented_img.astype(np.uint8)

    def _sample_next_batch(self):
        if self._index + self._batch_size > len(self._sample):
            self._shuffle_data()
            self._index = 0

        sample = self._sample[self._index:(self._index + self._batch_size)]
        self._index += self._batch_size

        images_blob = []
        labels_blob = []

        for i in xrange(self._batch_size):
            image, label = self.data_sampler_.get_image(sample[i])

            if image is None:
                continue

            augmented_image = self._augment(image)

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

        self.brightness_ = layer_params['brightness'] if 'brightness' in layer_params else False
        if self.brightness_:
            self.min_pos_ = layer_params['min_pos'] if 'min_pos' in layer_params else 100.0
            self.pos_alpha_ = layer_params['pos_alpha'] if 'pos_alpha' in layer_params else [0.2, 1.5]
            self.pos_beta_ = layer_params['pos_beta'] if 'pos_beta' in layer_params else [-100.0, 50.0]
            self.neg_alpha_ = layer_params['neg_alpha'] if 'neg_alpha' in layer_params else [0.9, 1.5]
            self.neg_beta_ = layer_params['neg_beta'] if 'neg_beta' in layer_params else [-20.0, 50.0]

        self.dither_ = layer_params['dither'] if 'dither' in layer_params else False
        if self.dither_:
            self.max_factor_left_ = layer_params['max_factor_left'] if 'max_factor_left' in layer_params else 0.1
            self.max_factor_right_ = layer_params['max_factor_right'] if 'max_factor_right' in layer_params else 0.1
            self.max_factor_top_ = layer_params['max_factor_top'] if 'max_factor_top' in layer_params else 0.1
            self.max_factor_bottom_ = layer_params['max_factor_bottom'] if 'max_factor_bottom' in layer_params else 0.1
            assert 0.0 < self.max_factor_left_ < 1.0
            assert 0.0 < self.max_factor_right_ < 1.0
            assert 0.0 < self.max_factor_left_ + self.max_factor_right_ < 1.0
            assert 0.0 < self.max_factor_top_ < 1.0
            assert 0.0 < self.max_factor_bottom_ < 1.0
            assert 0.0 < self.max_factor_top_ + self.max_factor_bottom_ < 1.0

        self.erase_ = layer_params['erase'] if 'erase' in layer_params else False
        if self.erase_:
            self.erase_max_size_ = layer_params['erase_max_size'] if 'erase_max_size' in layer_params else 0.1
            self.erase_border_ = layer_params['erase_max_size'] if 'erase_max_size' in layer_params else [0.05, 0.95]
            assert 0.0 < self.erase_max_size_ < 1.0
            assert 0.0 < self.erase_border_[0] < self.erase_border_[1] < 1.0

        self.mirror_ = layer_params['mirror'] if 'mirror' in layer_params else False

        if layer_params['input_type'] == 'lmdb':
            assert 'lmdb_path' in layer_params
            data_sampler = SampleDataFromLmdb(layer_params['lmdb_path'])
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
        top[0].reshape(self._batch_size, 3, self._trg_height, self._trg_width)
        top[1].reshape(self._batch_size)
