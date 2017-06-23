#ifndef CAFFE_GRN_LAYER_HPP_
#define CAFFE_GRN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Global Response Normalization with L2 norm.
 */
template <typename Dtype>
class GRNLayer : public Layer<Dtype> {
 public:
  explicit GRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GRN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// bias is used to prevent from zero division
  Dtype bias_;
  /// sum_multiplier is used to carry out sum using BLAS 1 x ch x 1 x 1
  Blob<Dtype> sum_multiplier_;
  /// square result n X ch x h x w
  Blob<Dtype> square_;
  /// norm is an intermediate Blob to hold temporary results. n * 1 * h * w
  Blob<Dtype> norm_;
  /// temp_dot n * 1 * h * w
  Blob<Dtype> temp_dot_;
};


}  // namespace caffe

#endif  // CAFFE_GRN_LAYER_HPP_
