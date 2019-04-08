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
#include <cmath>
#include <vector>

#include "caffe/layers/label_specific_add_layer.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void LabelSpecificAddForward(
      const int n, const int dim, const Dtype* label,
      Dtype* top_data, Dtype bias) {
    CUDA_KERNEL_LOOP(index, n) {
      const int gt = static_cast<int>(label[index]);
      if (top_data[index * dim + gt] > bias) {
        top_data[index * dim + gt] -= bias;
      }
    }
  }

  template <typename Dtype>
  void LabelSpecificAddLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    const int num = bottom[0]->num();
    const int count = bottom[0]->count();
    const int dim = count / num;

    if (top[0] != bottom[0]) {
      caffe_copy(count, bottom_data, top_data);
    }

    if (!transform_test_ && this->phase_ == TEST) {
      return;
    }

    // NOLINT_NEXT_LINE(whitespace/operators)
    LabelSpecificAddForward<Dtype> << <CAFFE_GET_BLOCKS(num),
        CAFFE_CUDA_NUM_THREADS >> > (num, dim, label_data, top_data, bias_);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  void LabelSpecificAddLayer<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
          << " Layer cannot backpropagate to labels.";
    }

    if (top[0] != bottom[0] && propagate_down[0]) {
      const int count = bottom[0]->count();
      const Dtype* top_diff = top[0]->gpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

      caffe_copy(count, top_diff, bottom_diff);
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(LabelSpecificAddLayer);
}  // namespace caffe
