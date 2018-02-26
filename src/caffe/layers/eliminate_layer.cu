#include <cfloat>
#include <vector>

#include "caffe/layers/eliminate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EliminateLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();

  caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top_data);

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int inner_dim = bottom[1]->count() / num;

  Dtype* off_data = off_values_.mutable_gpu_data();
  caffe_gpu_set(bottom[1]->count(), Dtype(off_value_), off_data);
  caffe_gpu_axpby(bottom[1]->count(), Dtype(-off_value_), bottom[1]->gpu_data(),
                  Dtype(1), off_data);

  const Dtype* mask_data = bottom[1]->gpu_data();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      caffe_gpu_mul(inner_dim, top_data, mask_data, top_data);
      caffe_gpu_axpby(inner_dim, Dtype(1), off_data, Dtype(1), top_data);

      top_data += inner_dim;
    }
    mask_data += inner_dim;
  }
}

template <typename Dtype>
void EliminateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(EliminateLayer);

}  // namespace caffe
