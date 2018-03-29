import numpy as np
import yaml
import caffe


class PushLossLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        self.margin_ = float(layer_params['margin']) if 'margin' in layer_params else 0.2

    def setup(self, bottom, top):
        self._load_params(self.param_str)

    def forward(self, bottom, top):
        num_bottoms = len(bottom)
        assert num_bottoms in [1, 2]

        embeddings = np.array(bottom[0].data).astype(np.float32)
        if len(bottom) == 2:
            labels = np.array(bottom[1].data).astype(np.int32)
        else:
            labels = np.array(range(bottom[0].num), dtype=np.int32)

        assert len(embeddings.shape) == 2
        assert len(labels.shape) == 1
        assert embeddings.shape[0] == labels.shape[0]

        all_pairs = labels.reshape([-1, 1]) != labels.reshape([1, -1])

        distances = 1.0 - np.matmul(embeddings, np.transpose(embeddings))
        losses = self.margin_ - distances
        self.valid_pairs = (all_pairs * (losses > 0.0)).astype(np.float32)
        self.num_valid_pairs = np.sum(self.valid_pairs)

        if self.num_valid_pairs > 0.0:
            valid_losses = losses * self.valid_pairs

            loss = np.sum(valid_losses) / float(self.num_valid_pairs)
            top[0].data[...] = loss

            if len(top) == 2:
                min_inter_class_dist = np.min(distances[all_pairs])
                top[1].data[...] = min_inter_class_dist
        else:
            top[0].data[...] = 0.0

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            embeddings_diff = np.zeros(bottom[0].data.shape)

            if self.num_valid_pairs > 0.0:
                embeddings = np.array(bottom[0].data).astype(np.float32)

                factor = top[0].diff[0] / float(self.num_valid_pairs)
                for i in xrange(bottom[0].num):
                    for j in xrange(i + 1, bottom[0].num):
                        if self.valid_pairs[i, j]:
                            embeddings_diff[i] += factor * embeddings[j]
                            embeddings_diff[j] += factor * embeddings[i]

            bottom[0].diff[...] = embeddings_diff

    def reshape(self, bottom, top):
        top[0].reshape(1)

        if len(top) == 2:
            top[1].reshape(1)
