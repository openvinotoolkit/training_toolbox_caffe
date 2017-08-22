#include <vector>
#include <float.h>
#include "caffe/layers/region_yolo_layer.hpp"

namespace caffe {

template <typename Dtype>
void RegionYoloLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int start_axis = bottom[0]->CanonicalAxisIndex(
        this->layer_param_.flatten_param().axis());
    const int end_axis = bottom[0]->CanonicalAxisIndex(
        this->layer_param_.flatten_param().end_axis());
    vector<int> top_shape;
    for (int i = 0; i < start_axis; ++i) {
      top_shape.push_back(bottom[0]->shape(i));
    }
    const int flattened_dim = bottom[0]->count(start_axis, end_axis + 1);
    top_shape.push_back(flattened_dim);
    for (int i = end_axis + 1; i < bottom[0]->num_axes(); ++i) {
      top_shape.push_back(bottom[0]->shape(i));
    }
    top[0]->Reshape(top_shape);
    CHECK_EQ(top[0]->count(), bottom[0]->count());
}

template <typename Dtype>
void RegionYoloLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    RegionYoloParameter region_yolo_param = this->layer_param_.region_yolo_param();
    classes_ = region_yolo_param.classes();
    coords_ = region_yolo_param.coords();
    num_ = region_yolo_param.num();
}

template <typename Dtype>
void RegionYoloLayer<Dtype>::softmax(const Dtype *input, int classes, int stride, Dtype *output)
{
    Dtype largest = -FLT_MAX;
    for (int i = 0; i < classes; ++i){
        if (input[i*stride] > largest) largest = input[i*stride];
    }

    Dtype sum = 0;
    for (int i = 0; i < classes; ++i){
        Dtype e = exp(input[i*stride] - largest);
        sum += e;
        output[i*stride] = e;
    }

    for (int i = 0; i < classes; ++i){
        output[i * stride] /= sum;
    }
}

template <typename Dtype>
void RegionYoloLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  int batch = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int height = bottom[0]->shape(2);
  int width = bottom[0]->shape(3);

  caffe_copy(width*height*channels*batch, bottom_data, top_data);

  int inputs = height*width*num_*(classes_ + coords_ + 1);
  for (int b = 0; b < batch; ++b){
      for (int n = 0; n < num_; ++n){
          int index = entry_index(width, height, coords_, classes_, inputs, b, n*width*height, 0);
          for (int i = index; i < index + 2*width*height; ++i){
            top_data[i] = logistic_activate(top_data[i]);
          }

          index = entry_index(width, height, coords_, classes_, inputs, b, n * width * height, coords_);
          for (int i = index; i < index + width*height; ++i){
            top_data[i] = logistic_activate(top_data[i]);
          }
      }
  }

  int index = entry_index(width, height, coords_, classes_, inputs, 0, 0, coords_ + 1);

  int batchOffset = inputs / num_;
  int groups = width * height;
  int groupOffset = 1;
  int stride = width * height;
  for (int b = 0; b < batch*num_; ++b) {
    for (int g = 0; g < groups; ++g) {
      softmax(bottom_data + index + b * batchOffset + g * groupOffset,
              classes_, stride,
              top_data + index + b * batchOffset + g * groupOffset);
    }
  }

  return;
}

template <typename Dtype>
void RegionYoloLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RegionYoloLayer);
#endif

INSTANTIATE_CLASS(RegionYoloLayer);
REGISTER_LAYER_CLASS(RegionYolo);

}  // namespace caffe
