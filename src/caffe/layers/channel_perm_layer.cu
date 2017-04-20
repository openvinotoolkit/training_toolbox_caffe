#include <vector>

#include "caffe/layers/channel_perm_layer.hpp"

namespace caffe {

template <typename Dtype>
void ChannelPermutationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_common(bottom, top);
}

template <typename Dtype>
void ChannelPermutationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])  return;

  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelPermutationLayer);

}  // namespace caffe
