#include <cmath>
#include <vector>

#include "caffe/layers/general_sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype general_sigmoid(Dtype x, Dtype alpha) {
  return (1. - alpha) / (1. + exp(-x)) + alpha;
}

template <typename Dtype>
void GeneralSigmoidLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  alpha_ = this->layer_param_.general_sigmoid_param().alpha();
  CHECK_GE(alpha_, 0.0);
  CHECK_LE(alpha_, 1.0);
}

template <typename Dtype>
void GeneralSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = general_sigmoid(bottom_data[i], alpha_);
  }
}

template <typename Dtype>
void GeneralSigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const Dtype factor = 1.0 / (1.0 - alpha_);
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = factor * top_diff[i] * (sigmoid_x - alpha_) * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GeneralSigmoidLayer);
#endif

INSTANTIATE_CLASS(GeneralSigmoidLayer);
// REGISTER_LAYER_CLASS(GeneralSigmoidLayer);

}  // namespace caffe
