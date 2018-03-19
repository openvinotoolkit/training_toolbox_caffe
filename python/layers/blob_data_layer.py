import numpy as np
import yaml
import caffe


class BlobDataLayer(caffe.Layer):
    def _sample_next_batch(self):
        if self.iter_ >= self.max_num_iters:
            raise Exception('Reached the max number of iterations.')
        else:
            batch_ids = self.ids_[self.iter_ * self.batch_size_:(self.iter_ + 1) * self.batch_size_]
            self.iter_ += 1

            images_blob = self.blobs_[batch_ids]
            labels_blob = self.labels_[batch_ids]

        return images_blob, labels_blob

    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        assert 'batch_size' in layer_params
        assert 'channels' in layer_params
        assert 'height' in layer_params
        assert 'width' in layer_params

        self.batch_size_ = layer_params['batch_size']
        self.channels_ = layer_params['channels']
        self.height_ = layer_params['height']
        self.width_ = layer_params['width']

    def _reset_states(self):
        self.iter_ = 0

    def update_data(self, blobs, labels):
        assert len(blobs.shape) == 4
        assert len(labels.shape) == 1
        assert blobs.shape[0] == labels.shape[0]
        assert blobs.shape[1] == self.channels_
        assert blobs.shape[2] == self.height_
        assert blobs.shape[3] == self.width_

        self.blobs_ = blobs
        self.labels_ = labels

        self._reset_states()

    def update_indices(self, ids, shuffle=False):
        assert len(ids) % self.batch_size_ == 0

        self.ids_ = np.copy(ids)
        self.max_num_iters = len(self.ids_) / self.batch_size_

        if shuffle:
            np.random.shuffle(self.ids_)

    def setup(self, bottom, top):
        self._load_params(self.param_str)
        self._reset_states()

        self.ids_ = None
        self.max_num_iters = 0

    def forward(self, bottom, top):
        images_blob, labels_blob = self._sample_next_batch()

        top[0].data[...] = images_blob
        top[1].data[...] = labels_blob

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size_, self.channels_, self.height_, self.width_)
        top[1].reshape(self.batch_size_)
