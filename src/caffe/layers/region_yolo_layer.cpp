#include <vector>
#include <float.h>
#include "caffe/layers/region_yolo_layer.hpp"

namespace caffe {

template <typename Dtype>
void RegionYoloLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
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
Dtype RegionYoloLayer<Dtype>::logistic_activate(Dtype x){return 1./(1. + exp(-x));}

template <typename Dtype>
void RegionYoloLayer<Dtype>::activate_array(Dtype *x, const int n)
{
    for (int i = 0; i < n; ++i){
        x[i] = logistic_activate(x[i]);
    }
}

template <typename Dtype>
void RegionYoloLayer<Dtype>::softmax(const Dtype *input, int n, int stride, Dtype *output)
{
    Dtype largest = -FLT_MAX;
    for (int i = 0; i < n; ++i){
        if (input[i*stride] > largest) largest = input[i*stride];
    }

    Dtype sum = 0;
    for (int i = 0; i < n; ++i){
        Dtype e = exp(input[i*stride] - largest);
        sum += e;
        output[i*stride] = e;
    }

    for (int i = 0; i < n; ++i){
        output[i * stride] /= sum;
    }
}

template <typename Dtype>
void RegionYoloLayer<Dtype>::softmax_cpu(const Dtype *input, int n, int batch, int batch_offset,
                                     int groups, int group_offset, int stride,
                                     Dtype *output)
{
    for (int b = 0; b < batch; ++b){
        for (int g = 0; g < groups; ++g){
            softmax(input + b * batch_offset + g * group_offset,
                    n, stride,
                    output + b * batch_offset + g * group_offset);
        }
    }
}

template <typename Dtype>
int RegionYoloLayer<Dtype>::entry_index(int w, int h, int coords, int classes,
                                    int outputs, int batch, int location,
                                    int entry)
{
    int n = location / (w * h);
    int loc = location % (w * h);
    return batch * outputs + n * w * h*(coords + classes + 1) + entry * w * h + loc;
}

template <typename Dtype>
void RegionYoloLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  int batch = top[0]->shape(0);
  int channels = top[0]->shape(1);
  int width = top[0]->shape(2);
  int height = top[0]->shape(3);

  caffe_copy(width*height*channels*batch, bottom_data, top_data);

  int inputs = height*width*num_*(classes_ + coords_ + 1);
  for (int b = 0; b < batch; ++b){
      for (int n = 0; n < num_; ++n){
          int index = entry_index(width, height, coords_, classes_, inputs, b, n*width*height, 0);
          activate_array(top_data + index, 2*width*height);

          index = entry_index(width, height, coords_, classes_, inputs, b, n * width * height, coords_);
          activate_array(top_data + index, width*height);
      }
  }

  int index = entry_index(width, height, coords_, classes_, inputs, 0, 0, coords_ + 1);
  softmax_cpu(bottom_data + index, classes_, batch*num_, inputs / num_, width * height, 1, width * height, top_data + index);

  return;
}

template <typename Dtype>
void RegionYoloLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
}

#ifdef CPU_ONLY
STUB_GPU(RegionYoloLayer);
#endif

INSTANTIATE_CLASS(RegionYoloLayer);
REGISTER_LAYER_CLASS(RegionYolo);

}  // namespace caffe
