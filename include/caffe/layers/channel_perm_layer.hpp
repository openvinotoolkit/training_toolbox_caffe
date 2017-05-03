#ifndef CAFFE_CHANNEL_PERMUTATION_LAYER_HPP_
#define CAFFE_CHANNEL_PERMUTATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Takes one blob, rearranges its channels (or fills some of them
 *        with constant value) according to layer parameters, and outputs
 *        the result.
 */
template <typename Dtype>
class ChannelPermutationLayer : public Layer<Dtype> {
 public:
  explicit ChannelPermutationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ChannelPermutation"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C_0 \times H \times W) @f$
   *      the input @f$ x @f$, 2- or more-dimensional (4D in this example).
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C_1 \times H \times W) @f$ rearranged output @f$ y @f$,
   *      where dimension @f$ C_1 = @f$num\_output,
   *      and @f$ y_{n,c,i,j} = x_{n,C,i,j} @f$ if parameters contain an action
   *      with @p chan @f$=c@f$ and @p copy @f$=C@f$,
   *      or @f$ y_{n,c,i,j} = A @f$ for action
   *      with @p chan @f$=c@f$ and @p fill @f$=A@f$.
   *      In-place operation is supported.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  void Forward_common(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the input.
   *        <b>(Not implemented.)</b>
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *        respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing loss gradients @f$ \frac{\partial L}{\partial y_{n,c,i,j}} @f$
   *      with respect to the corresponding entries of the output blob @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1), into which the bottom gradient
   *        @f$ \frac{\partial E}{\partial x_{n,c,i,j}} @f$ is written.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // If we're actually operating in-place and without temporary storage
  bool really_inplace_;
  // If we're using temporary storage to store the result.
  bool use_temp_storage_;
  // Blob for temporary storage of the result.
  Blob<Dtype> temp_;
};

}  // namespace caffe

#endif  // CAFFE_CHANNEL_PERMUTATION_LAYER_HPP_
