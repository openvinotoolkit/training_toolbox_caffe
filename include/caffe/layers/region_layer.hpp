#ifndef CAFFE_REGION_LAYER_HPP_
#define CAFFE_REGION_LAYER_HPP_

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <fstream>

namespace caffe {

/**
 * @brief Copy a Blob along specified dimensions.
 */
template <typename Dtype>
class RegionLayer : public Layer<Dtype> {
 public:
  explicit RegionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
    }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
private:
  Dtype logistic_activate(Dtype x);

  void activate_array(Dtype *x, const int n);

  void softmax(const Dtype *input, int n, int stride, Dtype *output);

  void softmax_cpu(const Dtype *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, Dtype *output);

  int entry_index(int w, int h, int coords, int classes, int outputs, int batch, int location, int entry);
};

}  // namespace caffe

#endif  // CAFFE_TILE_LAYER_HPP_
