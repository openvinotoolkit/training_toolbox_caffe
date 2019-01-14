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
#include <cfloat>
#include <vector>

#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PermuteKernel(const int nthreads,
    Dtype* const bottom_data, const bool forward, const int* permute_order,
    const int* old_steps, const int* new_steps, const int num_axes,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp_idx = index;
    int old_idx = 0;
    for (int i = 0; i < num_axes; ++i) {
      int order = permute_order[i];
      old_idx += (temp_idx / new_steps[i]) * old_steps[order];
      temp_idx %= new_steps[i];
    }
    if (forward) {
      top_data[index] = bottom_data[old_idx];
    } else {
      bottom_data[old_idx] = top_data[index];
    }
  }
}

template <typename Dtype>
void PermuteLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (need_permute_) {
    Dtype* bottom_data = bottom[0]->mutable_gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();
    const int* permute_order = permute_order_.gpu_data();
    const int* new_steps = new_steps_.gpu_data();
    const int* old_steps = old_steps_.gpu_data();
    bool foward = true;
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, foward, permute_order, old_steps, new_steps,
        num_axes_, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    // If there is no need to permute, we share data to save memory.
    top[0]->ShareData(*bottom[0]);
  }
}


template <typename Dtype>
void PermuteLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (need_permute_) {
    Dtype* top_diff = top[0]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const int* permute_order = permute_order_.gpu_data();
    const int* new_steps = new_steps_.gpu_data();
    const int* old_steps = old_steps_.gpu_data();
    bool foward = false;
    // NOLINT_NEXT_LINE(whitespace/operators)
    PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_diff, foward, permute_order, old_steps, new_steps,
        num_axes_, top_diff);
    CUDA_POST_KERNEL_CHECK;
  } else {
    // If there is no need to permute, we share diff to save memory.
    bottom[0]->ShareDiff(*top[0]);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PermuteLayer);

}  // namespace caffe
