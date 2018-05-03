import numpy as np
import caffe


class CentersStatLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def forward(self, bottom, top):
        assert len(bottom) == 1

        centers = np.array(bottom[0].data).astype(np.float32)
        assert len(centers.shape) == 2

        distances = 1. - np.matmul(centers, np.transpose(centers))
        values = distances[np.triu_indices_from(distances, k=1)]

        top[0].data[...] = np.min(values)
        top[1].data[...] = np.mean(values)
        top[2].data[...] = np.max(values)
        top[3].data[...] = np.std(values)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        top[3].reshape(1)
