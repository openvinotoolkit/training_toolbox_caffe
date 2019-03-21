/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <algorithm>
#include <vector>

#include "caffe/layers/grn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  GRNParameter param = this->layer_param().grn_param();
  bias_ = param.bias();
  CHECK_GE(bias_, 0.0) << "Bias must be >=0.";
  dummy_backward_ = param.dummy_backward();
}

template <typename Dtype>
void GRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  sum_multiplier_.Reshape(1, bottom[0]->channels(), 1, 1);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), static_cast<Dtype>(1), multiplier_data);
  square_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  temp_dot_.Reshape(bottom[0]->num(), 1,
                    bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void GRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* square_data = square_.mutable_cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  caffe_copy(bottom[0]->count(), bottom_data, square_data);
  // do the normalization.
  for (int i = 0; i < num; ++i) {
    // square each element
    caffe_sqr<Dtype>(dim, square_data + i * dim, square_data + i * dim);
    // sum cross the channel
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1.,
        square_data + i * dim, sum_multiplier_.cpu_data(), 0.,
        norm_data + i * spatial_dim);
    // add bias to prevent zero division
    caffe_add_scalar<Dtype>(spatial_dim, bias_, norm_data + i * spatial_dim);
    // root the square norm_data
    caffe_powx<Dtype>(spatial_dim, norm_data + i * spatial_dim, 0.5,
        norm_data + i * spatial_dim);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(spatial_dim, top_data + top[0]->offset(i, j),
          norm_data + i * spatial_dim, top_data + top[0]->offset(i, j));
    }
  }
}

template <typename Dtype>
void GRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_copy(top[0]->count(), top_diff, bottom_diff);

  if (dummy_backward_) {
    return;
  }

  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();
  Dtype* temp_dot_data = temp_dot_.mutable_cpu_data();
  Dtype* temp_data = square_.mutable_cpu_data();  // just reuse the square_
  int num = top[0]->num();
  int channels = top[0]->channels();
  int dim = top[0]->count() / top[0]->num();
  int spatial_dim = top[0]->height() * top[0]->width();

  for (int i = 0; i < num; ++i) {
    for (int k = 0; k < spatial_dim; ++k) {
      temp_dot_data[i * spatial_dim + k] = caffe_cpu_strided_dot<Dtype>(
          channels, top_diff + i * dim + k, spatial_dim,
          top_data + i * dim + k, spatial_dim)  /
          (norm_data[i * spatial_dim + k] * norm_data[i * spatial_dim + k]);
    }
    for (int j = 0; j < channels; j++) {
      caffe_div(spatial_dim, bottom_diff + top[0]->offset(i, j),
          norm_data + i * spatial_dim, bottom_diff + top[0]->offset(i, j));
      caffe_mul(spatial_dim, bottom_data + top[0]->offset(i, j),
          temp_dot_data + i * spatial_dim, temp_data + top[0]->offset(i, j));
      caffe_axpy(spatial_dim, Dtype(-1.0), temp_data + top[0]->offset(i, j),
          bottom_diff + top[0]->offset(i, j));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GRNLayer);
#endif

INSTANTIATE_CLASS(GRNLayer);
REGISTER_LAYER_CLASS(GRN);


}  // namespace caffe
