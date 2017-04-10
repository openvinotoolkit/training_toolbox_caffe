#include <vector>

#include "caffe/layers/channel_perm_layer.hpp"

namespace caffe {

template <typename Dtype>
void ChannelPermutationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  this->Forward_cpu(bottom, top);
}

template <typename Dtype>
void ChannelPermutationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  this->Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelPermutationLayer);

}  // namespace caffe
