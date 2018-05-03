import numpy as np
import caffe


class BatchDistLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def forward(self, bottom, top):
        embeddings = np.array(bottom[0].data)
        assert len(embeddings.shape) == 2

        distances = 1.0 - np.matmul(embeddings, embeddings.transpose())
        top[0].data[...] = distances

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            embeddings = np.array(bottom[0].data)
            bottom[0].diff[...] = -2.0 * np.sum(embeddings, axis=0, keepdims=True)

    def reshape(self, bottom, top):
        top[0].reshape((1, 1, bottom[0].num, bottom[0].num))
