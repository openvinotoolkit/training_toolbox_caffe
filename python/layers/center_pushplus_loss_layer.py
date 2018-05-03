import numpy as np
import yaml
import caffe


class CenterPushPlusLossLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        self.min_margin_ = float(layer_params['min_margin']) if 'min_margin' in layer_params else 0.5
        self.max_margin_ = float(layer_params['max_margin']) if 'max_margin' in layer_params else 0.6
        self.num_iter_ = float(layer_params['num_iter']) if 'num_iter' in layer_params else 10000

    def _init_states(self):
        self.num_additional_tops_ = 6
        self.iter = 0

    def setup(self, bottom, top):
        self._load_params(self.param_str)
        self._init_states()

    def forward(self, bottom, top):
        assert len(bottom) == 1

        centers = np.array(bottom[0].data).astype(np.float32)
        assert len(centers.shape) == 2

        num_centers = centers.shape[0]
        assert num_centers > 2

        neg_distances = 1.0 - np.matmul(centers, np.transpose(centers))

        margin = self.min_margin_ + float(self.iter) / float(self.num_iter_) * (self.max_margin_ - self.min_margin_)
        margin = np.minimum(margin, self.max_margin_)

        center_ids = np.arange(num_centers, dtype=np.int32)
        valid_pairs_mask = center_ids.reshape([-1, 1]) != center_ids.reshape([1, -1])
        valid_pairs_mask *= np.tri(*valid_pairs_mask.shape, k=-1, dtype=np.bool)

        losses = margin - neg_distances

        self.valid_mask = valid_pairs_mask * (losses > 0.0)
        self.num_valid_triplets = np.sum(self.valid_mask)

        loss = np.sum(losses) / float(self.num_valid_pairs)
        top[0].data[...] = loss

        if int(self.num_valid_triplets) > 0:
            valid_losses = losses * self.valid_mask.astype(np.float32)

            loss = np.sum(valid_losses) / float(self.num_valid_triplets)
            top[0].data[...] = loss

            if len(top) > 1:
                assert len(top) == self.num_additional_tops_ + 1

                top[1].data[...] = margin
                top[2].data[...] = float(self.num_valid_pairs) / float(np.maximum(1, np.sum(valid_pairs_mask)))

                all_valid_neg_distances = neg_distances[valid_pairs_mask]
                top[3].data[...] = np.min(all_valid_neg_distances)
                top[4].data[...] = np.mean(all_valid_neg_distances)
                top[5].data[...] = np.max(all_valid_neg_distances)
                top[6].data[...] = np.std(all_valid_neg_distances)
        else:
            top[0].data[...] = 0.0

        self.iter += 1

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            centers_diff = np.zeros(bottom[0].data.shape)

            if int(self.num_valid_triplets) > 0:
                centers = np.array(bottom[0].data).astype(np.float32)

                factor = top[0].diff[0] / float(self.num_valid_triplets)
                for i in xrange(centers.shape[0]):
                    for j in xrange(centers.shape[0]):
                        if self.valid_mask[i, j]:
                            centers_diff[i] += factor * centers[j]
                            centers_diff[j] += factor * centers[i]

            bottom[0].diff[...] = centers_diff

    def reshape(self, bottom, top):
        top[0].reshape(1)

        if len(top) > 1:
            assert len(top) == self.num_additional_tops_ + 1

            for i in xrange(self.num_additional_tops_):
                top[i + 1].reshape(1)
