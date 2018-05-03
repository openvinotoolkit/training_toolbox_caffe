import numpy as np
import caffe
import yaml


class ExtraCenterLossLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        self.oversampling_ = int(layer_params['oversampling_']) if 'oversampling_' in layer_params else 3
        assert self.oversampling_ > 0

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
        assert self.oversampling_ * embeddings.shape[1] == centers.shape[1]
        assert embeddings.shape[0] == labels.shape[0]

        batch_size = bottom[0].num
        num_centers = bottom[2].num

        self.sample_id_by_center_id = {}
        self.center_id_by_sample_id = {}
        sum_intra_class_dist = 0.0
        max_intra_class_dist = 0.0
        for sample_id in xrange(batch_size):
            label = labels[sample_id]
            assert 0 <= label < num_centers

            start_center_id = label * self.oversampling_
            end_center_id = (label + 1) * self.oversampling_
            distances = 1.0 - np.matmul(embeddings[sample_id].reshape([1, -1]),
                                        np.transpose(centers[start_center_id:end_center_id]))

            local_best_center_id = np.argmin(distances.reshape([-1]))
            dist_to_center = distances[local_best_center_id]

            sum_intra_class_dist += dist_to_center
            max_intra_class_dist = max(max_intra_class_dist, dist_to_center)

            best_center_id = start_center_id + local_best_center_id
            self.sample_id_by_center_id[best_center_id] =\
                self.sample_id_by_center_id.get(best_center_id, []) + [sample_id]
            self.center_id_by_sample_id[sample_id] = best_center_id

        loss = sum_intra_class_dist / float(batch_size)
        top[0].data[...] = loss

        if len(top) == 2:
            top[1].data[...] = max_intra_class_dist

    def backward(self, top, propagate_down, bottom):
        embeddings = np.array(bottom[0].data).astype(np.float32)
        centers = np.array(bottom[2].data).astype(np.float32)

        batch_size = bottom[0].num
        factor = top[0].diff[0] / float(batch_size)

        if propagate_down[0]:
            embeddings_diff = np.zeros(bottom[0].data.shape)
            for sample_id in self.center_id_by_sample_id.keys():
                center_id = self.center_id_by_sample_id[sample_id]
                embeddings_diff[sample_id] += -factor * centers[center_id]
            bottom[0].diff[...] = embeddings_diff

        if propagate_down[1]:
            raise Exception('Cannot propagate down through the labels')

        if propagate_down[2]:
            centers_diff = np.zeros(bottom[2].data.shape)
            for label in self.sample_id_by_center_id.keys():
                for sample_id in self.sample_id_by_center_id[label]:
                    centers_diff[label] += -factor * embeddings[sample_id]
            bottom[2].diff[...] += centers_diff

    def reshape(self, bottom, top):
        top[0].reshape(1)

        if len(top) == 2:
            top[1].reshape(1)
