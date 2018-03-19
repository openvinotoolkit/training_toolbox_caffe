import numpy as np
import caffe


class CrossEntropyLossLayer(caffe.Layer):
    def _init_states(self):
        self.eps = 1e-6
        self.log_trg_dist = None

    def setup(self, bottom, top):
        self._init_states()

    def forward(self, bottom, top):
        src_dist = np.array(bottom[0].data)
        trg_dist = np.array(bottom[1].data)
        num_samples = bottom[0].num

        assert len(src_dist.shape) == 2
        assert len(trg_dist.shape) == 2
        assert src_dist.shape[0] == trg_dist.shape[0]
        assert src_dist.shape[1] == trg_dist.shape[1]

        self.neg_log_trg_dist = np.negative(np.log(trg_dist + self.eps))
        dist = src_dist * self.neg_log_trg_dist

        loss = np.sum(dist) / float(num_samples)
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        src_dist = np.array(bottom[0].data)
        trg_dist = np.array(bottom[1].data)

        factor = top[0].diff[0] / float(bottom[0].num)

        if propagate_down[0]:
            bottom[0].diff[...] = factor * self.neg_log_trg_dist

        if propagate_down[1]:
            bottom[1].diff[...] = factor * np.negative(src_dist) / (trg_dist + self.eps)

    def reshape(self, bottom, top):
        top[0].reshape(1)
