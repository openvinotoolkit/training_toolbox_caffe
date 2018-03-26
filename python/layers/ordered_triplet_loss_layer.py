import numpy as np
import yaml
import caffe


class OrderedTripletLossLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        self.margin_ = layer_params['margin'] if 'margin' in layer_params else 0.2

    def _init_states(self):
        self.triplets = []

    def setup(self, bottom, top):
        self._load_params(self.param_str)
        self._init_states()

    def forward(self, bottom, top):
        bottom_data = np.array(bottom[0].data).astype(np.float32)
        num_samples = bottom[0].num

        assert len(bottom_data.shape) == 2
        assert num_samples % 3 == 0

        num_triplets = int(num_samples / 3)
        pos_distances = 1. - np.sum(bottom_data[:num_triplets] * bottom_data[num_triplets:(2 * num_triplets)], axis=0)
        neg_distances = 1. - np.sum(bottom_data[:num_triplets] * bottom_data[(2 * num_triplets):], axis=0)

        dist = 0.0
        intra_class_dist = 0.0
        inter_class_dist = 0.0

        self.triplets = []
        for triplet_id in xrange(num_triplets):
            anchor_id = triplet_id
            pos_id = num_triplets + triplet_id
            neg_id = 2 * num_triplets + triplet_id

            intra_class_dist += pos_distances[triplet_id]
            inter_class_dist += neg_distances[triplet_id]

            sample_dist = np.maximum(self.margin_ + pos_distances[triplet_id] - neg_distances[triplet_id], 0.0)
            dist += sample_dist
            self.triplets.append((anchor_id, pos_id, neg_id))

        loss = dist / float(num_triplets)
        top[0].data[...] = loss

        if len(top) == 3:
            top[1].data[...] = intra_class_dist / float(num_triplets)
            top[2].data[...] = inter_class_dist / float(num_triplets)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom_data = np.array(bottom[0].data)

            factor = top[0].diff[0] / float(bottom[0].num / 3)
            bottom_diff = np.zeros(bottom[0].data.shape)
            for anchor_id, pos_id, neg_id in self.triplets:
                bottom_diff[anchor_id] += factor * (bottom_data[neg_id] - bottom_data[pos_id])
                bottom_diff[pos_id] += -factor * bottom_data[anchor_id]
                bottom_diff[neg_id] += factor * bottom_data[anchor_id]

            bottom[0].diff[...] = bottom_diff

    def reshape(self, bottom, top):
        top[0].reshape(1)

        if len(top) == 3:
            top[1].reshape(1)
            top[2].reshape(1)
