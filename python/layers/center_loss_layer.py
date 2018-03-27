import numpy as np
import caffe


class CenterLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

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

        batch_size = bottom[0].num
        num_centers = bottom[2].num

        self.sample_id_by_label = {}
        intra_class_dist = 0.0
        for sample_id in xrange(batch_size):
            label = labels[sample_id]
            assert 0 <= label < num_centers

            self.sample_id_by_label[label] = self.sample_id_by_label.get(label, []) + [sample_id]

            intra_class_dist += 1.0 - np.sum(embeddings[sample_id] * centers[label])

        loss = intra_class_dist / float(batch_size)
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        embeddings = np.array(bottom[0].data).astype(np.float32)
        labels = np.array(bottom[1].data).astype(np.int32)
        centers = np.array(bottom[2].data).astype(np.float32)

        batch_size = bottom[0].num
        factor = 1.0 / float(batch_size)

        if propagate_down[0]:
            embeddings_diff = np.zeros(bottom[0].data.shape)
            for sample_id in xrange(batch_size):
                label = labels[sample_id]
                embeddings_diff[sample_id] = -factor * centers[label]
            bottom[0].diff[...] = embeddings_diff

        if propagate_down[1]:
            raise Exception('Cannot propagate down through the labels')

        if propagate_down[2]:
            centers_diff = np.zeros(bottom[2].data.shape)
            for label in self.sample_id_by_label.keys():
                for sample_id in self.sample_id_by_label[label]:
                    centers_diff[label] = -factor * embeddings[sample_id]
            bottom[2].diff[...] = centers_diff

    def reshape(self, bottom, top):
        top[0].reshape(1)
