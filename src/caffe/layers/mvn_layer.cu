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

#include <vector>

#include "caffe/layers/mvn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MVNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  // subtract mean
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());  // EX
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
      temp_.mutable_gpu_data());
  caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(),
      top_data);  // X-EX

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.gpu_data(),
        sum_multiplier_.gpu_data(), 0.,
        variance_.mutable_gpu_data());  // E((X-EX)^2)

    // normalize variance
    caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
          variance_.mutable_gpu_data());

    caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
          temp_.mutable_gpu_data());

    caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
  }
}

template <typename Dtype>
void MVNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
          sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
          bottom_diff);
    caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff,
            sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_.gpu_data(), sum_multiplier_.gpu_data(), 1.,
            bottom_diff);

    caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
        bottom_diff);

    // put the squares of bottom into temp_
    caffe_gpu_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_gpu_data());

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
        temp_.mutable_gpu_data());

    caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);
  } else {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, top_diff,
            sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
            temp_.mutable_gpu_data());
    caffe_gpu_add(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(MVNLayer);


}  // namespace caffe
