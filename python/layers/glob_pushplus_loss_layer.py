import numpy as np
import yaml
import caffe


class GlobPushLossLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        self.margin_ = float(layer_params['margin']) if 'margin' in layer_params else 0.2
        self.num_additional_tops_ =\
            int(layer_params['num_additional_tops']) if 'num_additional_tops' in layer_params else 8

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

        center_ids = np.arange(centers.shape[0], dtype=np.int32)
        same_class_pairs = labels.reshape([-1, 1]) == center_ids.reshape([1, -1])
        different_class_pairs = labels.reshape([-1, 1]) != center_ids.reshape([1, -1])

        neg_distances = 1.0 - np.matmul(embeddings, np.transpose(centers))
        pos_distances = 1.0 - np.sum(embeddings * centers[labels], axis=1)
        losses = self.margin_ + pos_distances.reshape([-1, 1]) - neg_distances

        self.valid_mask = different_class_pairs * (losses > 0.0)
        self.num_valid_triplets = np.sum(self.valid_mask)

        if int(self.num_valid_triplets) > 0:
            valid_losses = losses * self.valid_mask.astype(np.float32)

            loss = np.sum(valid_losses) / float(self.num_valid_triplets)
            top[0].data[...] = loss

            if len(top) > 1:
                assert len(top) == self.num_additional_tops_ + 1

                all_valid_distances = neg_distances[different_class_pairs]
                top[1].data[...] = np.min(all_valid_distances)  # min_inter_class_dist
                top[2].data[...] = np.mean(all_valid_distances)  # mean_inter_class_dist
                top[3].data[...] = np.max(all_valid_distances)  # max_inter_class_dist
                top[4].data[...] = np.std(all_valid_distances)  # std_inter_class_dist

                same_class_dist = neg_distances[same_class_pairs]
                top[5].data[...] = np.min(same_class_dist)  # min_intra_class_dist
                top[6].data[...] = np.mean(same_class_dist)  # mean_intra_class_dist
                top[7].data[...] = np.max(same_class_dist)  # max_intra_class_dist
                top[8].data[...] = np.std(same_class_dist)  # std_intra_class_dist
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
                for sample_id in xrange(embeddings.shape[0]):
                    label = labels[sample_id]
                    for center_id in xrange(centers.shape[0]):
                        if self.valid_mask[sample_id, center_id]:
                            embeddings_diff[sample_id] += factor * (centers[center_id] - centers[label])

            bottom[0].diff[...] = embeddings_diff

    def reshape(self, bottom, top):
        top[0].reshape(1)

        if len(top) > 1:
            assert len(top) == self.num_additional_tops_ + 1

            for i in xrange(self.num_additional_tops_):
                top[i + 1].reshape(1)
