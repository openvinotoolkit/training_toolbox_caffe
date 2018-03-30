#include <cmath>
#include <vector>

#include "caffe/layers/general_sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GeneralSigmoidForward(const int n, Dtype alpha,
                                      const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = (1. - alpha) / (1. + exp(-in[index])) + alpha;
  }
}

template <typename Dtype>
void GeneralSigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  GeneralSigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, alpha_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void GeneralSigmoidBackward(const int n, Dtype alpha,
                                       const Dtype* in_diff,
                                       const Dtype* out_data, Dtype* out_diff) {
  Dtype factor = 1. / (1. - alpha);

  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = factor * in_diff[index] * (sigmoid_x - alpha) * (1 - sigmoid_x);
  }
}

template <typename Dtype>
void GeneralSigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    GeneralSigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, alpha_, top_diff, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GeneralSigmoidLayer);


}  // namespace caffe
