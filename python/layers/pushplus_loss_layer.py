import numpy as np
import yaml
import caffe


class PushPlusLossLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        self.min_margin_ = float(layer_params['min_margin']) if 'min_margin' in layer_params else 0.15
        self.max_margin_ = float(layer_params['max_margin']) if 'max_margin' in layer_params else 0.6
        self.num_iter_ = float(layer_params['num_iter']) if 'num_iter' in layer_params else 40000

    def _init_states(self):
        self.num_additional_tops_ = 6
        self.iter = 0

    def setup(self, bottom, top):
        self._load_params(self.param_str)
        self._init_states()

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

        margin = self.min_margin_ + float(self.iter) / float(self.num_iter_) * (self.max_margin_ - self.min_margin_)
        margin = np.minimum(margin, self.max_margin_)

        losses = margin + pos_distances.reshape([-1, 1]) - neg_distances
        self.valid_mask = all_pairs * (losses > 0.0)
        self.num_valid_triplets = np.sum(self.valid_mask)

        if int(self.num_valid_triplets) > 0:
            valid_losses = losses * self.valid_mask.astype(np.float32)

            loss = np.sum(valid_losses) / float(self.num_valid_triplets)
            top[0].data[...] = loss

            if len(top) > 1:
                assert len(top) == self.num_additional_tops_ + 1

                top[1].data[...] = margin
                top[2].data[...] = float(self.num_valid_triplets) / float(np.maximum(1, np.sum(all_pairs)))

                all_valid_neg_distances = neg_distances[all_pairs]
                top[3].data[...] = np.min(all_valid_neg_distances)   # min_inter_class_dist
                top[4].data[...] = np.mean(all_valid_neg_distances)  # mean_inter_class_dist
                top[5].data[...] = np.max(all_valid_neg_distances)   # max_inter_class_dist
                top[6].data[...] = np.std(all_valid_neg_distances)   # std_inter_class_dist
        else:
            top[0].data[...] = 0.0

        self.iter += 1

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
                    anchor_center = centers[label]

                    for ref_id in xrange(embeddings.shape[0]):
                        if self.valid_mask[anchor_id, ref_id]:
                            embeddings_diff[anchor_id] += factor * (embeddings[ref_id] - anchor_center)

            bottom[0].diff[...] = embeddings_diff

    def reshape(self, bottom, top):
        top[0].reshape(1)

        if len(top) > 1:
            assert len(top) == self.num_additional_tops_ + 1

            for i in xrange(self.num_additional_tops_):
                top[i + 1].reshape(1)
