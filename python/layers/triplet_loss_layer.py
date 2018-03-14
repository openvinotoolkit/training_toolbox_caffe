import numpy as np
import yaml
import caffe


class TripletLossLayer(caffe.Layer):
    def _load_params(self, param_str):
        layer_params = yaml.load(param_str)

        self.margin_ = layer_params['margin'] if 'margin' in layer_params else 0.2

    @staticmethod
    def _parse_labels(labels):
        data_table = {}
        for i in xrange(len(labels)):
            label = labels[i]
            data_table[label] = data_table.get(label, []) + [i]
        return data_table

    def _init_states(self):
        self.triplets = []

    def setup(self, bottom, top):
        self._load_params(self.param_str)
        self._init_states()

    def forward(self, bottom, top):
        bottom_data = np.array(bottom[0].data)
        bottom_label = np.array(bottom[1].data)
        num_samples = bottom[0].num

        assert len(bottom_data.shape) == 2
        assert len(bottom_label.shape) == 1
        assert bottom_data.shape[0] == bottom_label.shape[0]

        ids_by_label_map = self._parse_labels(bottom_label)
        labels = ids_by_label_map.keys()

        distances = 1.0 - np.matmul(bottom_data, bottom_data.transpose())

        dist = 0.0
        self.triplets = []
        for anchor_label in labels:
            local_ids = ids_by_label_map[anchor_label]
            for anchor_id in local_ids:
                positive_ids = [i for i in local_ids if i != anchor_id]
                if len(positive_ids) <= 0:
                    continue
                positive_dist = [distances[anchor_id, i] for i in positive_ids]
                pos_id = positive_ids[np.argmax(positive_dist)]

                negative_ids = [i for i in xrange(num_samples) if i not in positive_ids]
                if len(negative_ids) <= 0:
                    continue
                negative_dist = [distances[anchor_id, i] for i in negative_ids]
                neg_id = negative_ids[np.argmin(negative_dist)]

                sample_dist = self.margin_ + distances[anchor_id, pos_id] - distances[anchor_id, neg_id]
                dist += np.maximum(sample_dist, 0.0)

                self.triplets.append((anchor_id, pos_id, neg_id))

        loss = dist / float(num_samples)
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom_data = np.array(bottom[0].data)

            factor = top[0].diff[0] / float(bottom[0].num)
            bottom_diff = np.zeros(bottom[0].data.shape)
            for anchor_id, pos_id, neg_id in self.triplets:
                bottom_diff[anchor_id] += factor * (bottom_data[neg_id] - bottom_data[pos_id])
                bottom_diff[pos_id] += -factor * bottom_data[anchor_id]
                bottom_diff[neg_id] += factor * bottom_data[anchor_id]

            bottom[0].diff[...] = bottom_diff

    def reshape(self, bottom, top):
        top[0].reshape(1)
