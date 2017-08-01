#ifndef CAFFE_REGION_LAYER_HPP_
#define CAFFE_REGION_LAYER_HPP_

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Copy a Blob along specified dimensions.
 */
template <typename Dtype>
class RegionLayer : public Layer<Dtype> {
 public:
  explicit RegionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Region"; }

 protected:
 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);
 virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top);

 virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int coords_, classes_, num_;
  int log_, sqrt_;

  int softmax_, background_, max_;
  float jitter_;
  int rescore_;

  float thresh_;
  int classfix_, absolute_;
  float random_;

  float coord_scale_, object_scale_, noobject_scale_, mask_scale_, class_scale_;
  int bias_match_;
  std::string anchors_;
};

}  // namespace caffe

#endif  // CAFFE_TILE_LAYER_HPP_
