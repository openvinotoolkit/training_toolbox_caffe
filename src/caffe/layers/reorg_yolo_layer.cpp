#include <vector>

#include "caffe/layers/reorg_yolo_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReorgYoloLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    vector<int> top_shape;
    top_shape.push_back(bottom[0]->shape(0));
    top_shape.push_back(bottom[0]->shape(1) * stride_ * stride_);
    top_shape.push_back(bottom[0]->shape(2) / stride_);
    top_shape.push_back(bottom[0]->shape(3) / stride_);

    top[0]->Reshape(top_shape);
    CHECK_EQ(top[0]->count(), bottom[0]->count());

    return;
}

template <typename Dtype>
void ReorgYoloLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    ReorgYoloParameter reorg_yolo_param = this->layer_param_.reorg_yolo_param();
    stride_ = reorg_yolo_param.stride();
}


template <typename Dtype>
void ReorgYoloLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int batch = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);

  int out_c = channels / (stride_*stride_);
  for (int b = 0; b < batch; ++b) {
      for (int c = 0; c < channels; ++c) {
          for (int y = 0; y < height; ++y) {
              for(int x = 0; x < width; ++x) {
                  int in_index  = x + width*y + width*height*c + width*height*channels*b;

                  int c2 = c % out_c;
                  int offset = c / out_c;

                  int w2 = x*stride_ + offset % stride_;
                  int h2 = y*stride_ + offset / stride_;

                  int out_index = w2 + width*stride_*h2 + width*stride_*height*stride_*c2 + width*stride_*height*stride_*out_c*b;
                  top_data[in_index] = bottom_data[out_index];
              }
          }
      }
  }
  return;
}

template <typename Dtype>
void ReorgYoloLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ReorgYoloLayer);
#endif

INSTANTIATE_CLASS(ReorgYoloLayer);
REGISTER_LAYER_CLASS(ReorgYolo);

}  // namespace caffe
