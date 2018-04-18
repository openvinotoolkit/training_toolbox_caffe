import numpy as np
import yaml
import caffe


class GlobPushPlusDistLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        assert 'batch_size' in layer_params

        self.batch_size_ = layer_params['batch_size']
        self.margin_ = float(layer_params['margin']) if 'margin' in layer_params else 0.6

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

        center_ids = np.arange(centers.shape[0], dtype=np.int32)
        same_class_pairs = labels.reshape([-1, 1]) == center_ids.reshape([1, -1])

        neg_distances = 1.0 - np.matmul(embeddings, np.transpose(centers))
        pos_distances = 1.0 - np.sum(embeddings * centers[labels], axis=1)
        losses = self.margin_ + pos_distances.reshape([-1, 1]) - neg_distances

        invalid_mask = same_class_pairs * (losses <= 0.0)
        losses[invalid_mask] = 0.0

        max_losses = np.max(losses, axis=1)
        top[0].data[...] = max_losses

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size_)
