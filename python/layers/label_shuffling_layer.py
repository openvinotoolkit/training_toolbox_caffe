from fileinput import filename

import yaml
import caffe
from caffe.io import caffe_pb2
import numpy as np
import lmdb
from tqdm import tqdm
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter


def parseRecord(val):
    datum = caffe_pb2.Datum.FromString(val)
    data = caffe.io.datum_to_array(datum)
    label = np.array([datum.label])

    return data, label


def initialSetup(path):
    keys_labels = dict()
    labels_keys = dict()
    env = lmdb.open(path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, value in tqdm(cursor, desc='label_shuffling_layer initialSetup', total=env.stat()['entries']):
            datum = caffe_pb2.Datum.FromString(value)
            arr = caffe.io.datum_to_array(datum)
            label = datum.label;
            if not labels_keys.has_key(label):
                labels_keys[label] = []
            labels_keys[label].append(k)
            keys_labels[k] = label

    return keys_labels, labels_keys


# generate batch with predefined number of different classes and images per class
class LabelShufflingLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)

        self.source_ = layer_params['source']
        self.scales_ = layer_params['scales']
        self.subtract_ = np.zeros_like(self.scales_)
        if 'subtract' in layer_params:
            self.subtract_ = layer_params['subtract']

        self.batch_size_ = layer_params['batch_size']  # self.num_labels_*self.images_per_label_

        self.max_number_object_per_label_ = np.inf
        if 'max_number_object_per_label' in layer_params:
            self.max_number_object_per_label_ = layer_params['max_number_object_per_label']

        self.same_label_in_batch_ = 3
        if 'same_label_in_batch' in layer_params:
            self.same_label_in_batch_ = layer_params['same_label_in_batch']

        self.mirror_ = False
        if 'mirror' in layer_params:
            self.mirror_ = layer_params['mirror']

        self.dither_ = False
        if 'dither' in layer_params:
            self.dither_ = layer_params['dither']

        self.blur_ = False
        if 'blur' in layer_params:
            self.blur_ = layer_params['blur']

        self.change_brightness_ = False
        if 'change_brightness' in layer_params:
            self.change_brightness_ = layer_params['change_brightness']

        # structures setup
        self.keys_labels_, self.labels_keys_ = initialSetup(self.source_)
        self.keys_ = self.keys_labels_.keys()
        self.labels_ = self.labels_keys_.keys()

        np.random.shuffle(self.labels_)

        self.label_index_ = 0
        self.image_index_ = 0

        # figure out the shape
        k = self.keys_[0]

        env = lmdb.open(self.source_, readonly=True)
        with env.begin() as txn:
            val = txn.get(k)
        data, label = parseRecord(val)
        self.data_shape_ = data.shape
        self.label_shape_ = label.shape
        print('self.data_shape_')
        print(self.data_shape_)


    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size_, self.data_shape_[2], self.data_shape_[0], self.data_shape_[1])
        top[1].reshape(self.batch_size_, self.label_shape_[0])

    def getNewLabel(self, labels_in_batch):
        while True:
            new_label = np.random.choice(self.labels_)

            if not new_label in labels_in_batch:
                return new_label

    def getNewKey(self, label, old_keys):
        while True:
            new_key = np.random.choice(self.labels_keys_[label])

            if not new_key in old_keys:
                return new_key

    def getBatch(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size_

        batch_keys = []
        labels_in_batch = []

        while len(batch_keys) < batch_size:
            new_label = self.getNewLabel(labels_in_batch)
            labels_in_batch.append(new_label)

            n_keys = min(self.same_label_in_batch_, batch_size - len(batch_keys))
            keys = np.random.permutation(self.labels_keys_[new_label])[:n_keys].tolist()

            batch_keys.extend(keys)

        return batch_keys

    def augment(self, img):
        augmented_img = img

        if self.blur_:
            rand = np.random.randint(0, 2)
            if rand == 1:
                filter_size = np.random.uniform(low=0.0, high=0.5)

                for i in range(3):
                    augmented_img[:, :, i] = gaussian_filter(augmented_img[:, :, i], sigma=filter_size)

        if self.dither_:
            width = self.data_shape_[1]
            height = self.data_shape_[0]

            min_factor = 0
            max_factor_left_right = 0.1
            max_factor_top = 0.1
            max_factor_bottom = 0.2

            distance_from_edge_left = int(np.random.uniform(min_factor, max_factor_left_right) * width)
            right_edge = int(width * (1 - np.random.uniform(min_factor, max_factor_left_right)))
            distance_from_edge_top = int(np.random.uniform(min_factor, max_factor_top) * height)
            bottom_edge = int(height * (1 - np.random.uniform(min_factor, max_factor_bottom)))

            crop = augmented_img[distance_from_edge_top : bottom_edge, distance_from_edge_left : right_edge, ::]
            augmented_img = imresize(crop, (height, width))

        if self.mirror_:
            rand = np.random.randint(0, 2)
            if rand == 1:
                augmented_img = augmented_img[:, ::-1, :]

        if self.change_brightness_:
            rand = np.random.randint(0, 2)
            if rand == 1:

                if np.average(augmented_img) > 100:
                    alpha = np.random.uniform(0.2, 1.5)
                    beta = np.random.randint(-100, 50)
                else:
                    alpha = np.random.uniform(0.9, 1.5)
                    beta = np.random.randint(-20, 50)

                changed_brightness = augmented_img * alpha + beta

                augmented_img = np.where(changed_brightness < 255,
                                         changed_brightness,
                                         np.full_like(augmented_img, 255, dtype=np.uint8))
                augmented_img = np.where(augmented_img >= 0,
                                         augmented_img,
                                         np.full_like(augmented_img, 0, dtype=np.uint8))

        return augmented_img

    def convert_to_caffe_layout(self, img):

        img = img.transpose((2, 0, 1))
        return img # w, h, c -> c, w, h

    def forward(self, bottom, top):
        imgs = []
        labels = []

        batch_keys = self.getBatch()
        env = lmdb.open(self.source_, readonly=True)

        with env.begin() as txn:
            for k in batch_keys:
                val = txn.get(k)
                fields = parseRecord(val)

                imgs.append(self.convert_to_caffe_layout(self.augment(fields[0])))
                labels.append(fields[1])

            top[0].data[...] = np.array(imgs)
            top[1].data[...] = np.array(labels)

    def backward(self, top, propagate_down, bottom):
        pass
