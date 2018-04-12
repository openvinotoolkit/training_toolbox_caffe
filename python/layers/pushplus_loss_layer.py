import numpy as np
import yaml
import caffe


class PushPlusLossLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        self.margin_ = float(layer_params['margin']) if 'margin' in layer_params else 0.2

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

        all_pairs = labels.reshape([-1, 1]) != labels.reshape([1, -1])

        neg_distances = 1.0 - np.matmul(embeddings, np.transpose(embeddings))
        pos_distances = 1.0 - np.sum(embeddings * centers[labels], axis=1)

        losses = self.margin_ + pos_distances.reshape([-1, 1]) - neg_distances
        self.valid_mask = all_pairs * (losses > 0.0)
        self.num_valid_triplets = np.sum(self.valid_mask)

        if int(self.num_valid_triplets) > 0:
            valid_losses = losses * self.valid_mask.astype(np.float32)

            loss = np.sum(valid_losses) / float(self.num_valid_triplets)
            top[0].data[...] = loss

            if len(top) == 2:
                min_inter_class_dist = np.min(neg_distances[all_pairs])
                top[1].data[...] = min_inter_class_dist
        else:
            top[0].data[...] = 0.0

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            embeddings_diff = np.zeros(bottom[0].data.shape)

            if int(self.num_valid_triplets) > 0:
                embeddings = np.array(bottom[0].data).astype(np.float32)
                labels = np.array(bottom[1].data).astype(np.int32)
                centers = np.array(bottom[2].data).astype(np.float32)

                factor = top[0].diff[0] / float(self.num_valid_triplets)
                for anchor_id in xrange(embeddings.shape[0]):
                    label = labels[anchor_id]
                    for ref_id in xrange(centers.shape[0]):
                        if self.valid_mask[anchor_id, ref_id]:
                            embeddings_diff[anchor_id] += factor * (embeddings[ref_id] - centers[label])

            bottom[0].diff[...] = embeddings_diff

    def reshape(self, bottom, top):
        top[0].reshape(1)

        if len(top) == 2:
            top[1].reshape(1)
