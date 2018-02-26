#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/eliminate_layer.hpp"

namespace caffe {

template <typename Dtype>
void EliminateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  off_value_ = this->layer_param_.eliminate_param().off_value();

  CHECK_EQ(bottom[1]->channels(), 1);

  const int channels = bottom[0]->channels();
  CHECK_EQ(bottom[0]->count(), channels * bottom[1]->count());
}

template <typename Dtype>
void EliminateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  off_values_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void EliminateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();

  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top_data);

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int inner_dim = bottom[1]->count() / num;

  Dtype* off_data = off_values_.mutable_cpu_data();
  caffe_set(bottom[1]->count(), Dtype(off_value_), off_data);
  caffe_cpu_axpby(bottom[1]->count(), Dtype(-off_value_), bottom[1]->cpu_data(),
                  Dtype(1), off_data);

  const Dtype* mask_data = bottom[1]->cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      caffe_mul(inner_dim, top_data, mask_data, top_data);
      caffe_cpu_axpby(inner_dim, Dtype(1), off_data, Dtype(1), top_data);

      top_data += inner_dim;
    }
    mask_data += inner_dim;
  }
}

template <typename Dtype>
void EliminateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(EliminateLayer);
#endif

INSTANTIATE_CLASS(EliminateLayer);
REGISTER_LAYER_CLASS(Eliminate);

}  // namespace caffe
