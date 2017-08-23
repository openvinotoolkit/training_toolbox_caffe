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

  int out_channels = channels / (stride_*stride_);
  for (int ib = 0; ib < batch; ++ib) {
    for (int ic = 0; ic < channels; ++ic) {
      for (int iy = 0; iy < height; ++iy) {
        for (int ix = 0; ix < width; ++ix) {
          int inIndex  = ix + iy*width + ic*width*height + ib*width*height*channels;

          int oc = ic % out_channels;
          int offset = ic / out_channels;

          int ow = ix*stride_ + offset % stride_;
          int oh = iy*stride_ + offset / stride_;

          int outIndex = ow + oh*width*stride_ + oc*width*stride_*height*stride_ + ib*width*stride_*height*stride_*out_channels;
          top_data[inIndex] = bottom_data[outIndex];
        }
      }
    }
  }
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
