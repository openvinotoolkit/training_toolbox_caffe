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
 * @brief Region yolo
 */
template <typename Dtype>
class RegionYoloLayer : public Layer<Dtype> {
 public:
  explicit RegionYoloLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
    }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Region"; }
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

  int coords_, classes_, num_;
private:
  inline int entry_index(int width, int height, int coords, int classes, int outputs, int batch, int location, int entry) const {
      int n = location / (width * height);
      int loc = location % (width * height);
      return batch * outputs + n * width * height * (coords + classes + 1) + entry * width * height + loc;
  }

  inline Dtype logistic_activate(Dtype x) { return 1./(1. + exp(-x)); }

  void softmax(const Dtype *input, int n, int stride, Dtype *output);
};

}  // namespace caffe

#endif  // CAFFE_TILE_LAYER_HPP_
