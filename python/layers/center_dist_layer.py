import numpy as np
import yaml
import caffe


class CenterDistLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        assert 'batch_size' in layer_params
        self.batch_size_ = layer_params['batch_size']

        assert self.batch_size_ > 0

    def setup(self, bottom, top):
        self._load_params(self.param_str)

    def forward(self, bottom, top):
        assert len(bottom) == 3

        embeddings = np.array(bottom[0].data).astype(np.float32)
        labels = np.array(bottom[1].data).astype(np.int32)
        centers = np.array(bottom[2].data).astype(np.float32)

        assert len(embeddings.shape) == 2
        assert len(labels.shape) == 1
        assert len(centers.shape) == 2
        assert embeddings.shape[1] == centers.shape[1]
        assert embeddings.shape[0] == labels.shape[0]
        assert embeddings.shape[0] == self.batch_size_

        distances = 1.0 - np.sum(embeddings * centers[labels], axis=1)
        top[0].data[...] = distances

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size_)
