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

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // Special case where we can ignore the input: scale or power is 0.
  if (diff_scale_ == Dtype(0)) {
    Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
    caffe_gpu_set(count, value, top_data);
    return;
  }
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_copy(count, bottom_data, top_data);
  if (scale_ != Dtype(1)) {
    caffe_gpu_scal(count, scale_, top_data);
  }
  if (shift_ != Dtype(0)) {
    caffe_gpu_add_scalar(count, shift_, top_data);
  }
  if (power_ != Dtype(1)) {
    caffe_gpu_powx(count, top_data, power_, top_data);
  }
}

template <typename Dtype>
void PowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    if (diff_scale_ == Dtype(0) || power_ == Dtype(1)) {
      caffe_gpu_set(count, diff_scale_, bottom_diff);
    } else {
      const Dtype* bottom_data = bottom[0]->gpu_data();
      // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
      //               = diff_scale * y / (shift + scale * x)
      if (power_ == Dtype(2)) {
        // Special case for y = (shift + scale * x)^2
        //     -> dy/dx = 2 * scale * (shift + scale * x)
        //              = diff_scale * shift + diff_scale * scale * x
        caffe_gpu_axpby(count, diff_scale_ * scale_, bottom_data,
            Dtype(0), bottom_diff);
        if (shift_ != Dtype(0)) {
          caffe_gpu_add_scalar(count, diff_scale_ * shift_, bottom_diff);
        }
      } else if (shift_ == Dtype(0)) {
        // Special case for y = (scale * x)^power
        //     -> dy/dx = scale * power * (scale * x)^(power - 1)
        //              = scale * power * (scale * x)^power * (scale * x)^(-1)
        //              = power * y / x
        const Dtype* top_data = top[0]->gpu_data();
        caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        caffe_gpu_scal(count, power_, bottom_diff);
      } else {
        caffe_copy(count, bottom_data, bottom_diff);
        if (scale_ != Dtype(1)) {
          caffe_gpu_scal(count, scale_, bottom_diff);
        }
        if (shift_ != Dtype(0)) {
          caffe_gpu_add_scalar(count, shift_, bottom_diff);
        }
        const Dtype* top_data = top[0]->gpu_data();
        caffe_gpu_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
        if (diff_scale_ != Dtype(1)) {
          caffe_gpu_scal(count, diff_scale_, bottom_diff);
        }
      }
    }
    caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PowerLayer);


}  // namespace caffe
