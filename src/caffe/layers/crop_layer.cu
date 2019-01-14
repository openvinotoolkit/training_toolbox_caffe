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

#include "caffe/layers/crop_layer.hpp"

namespace caffe {

__device__ int compute_uncropped_index(
    int index,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets) {
  int dest_index = index;
  int src_index = 0;
  for (int i = 0; i < ndims; ++i) {
      int coord = dest_index / dest_strides[i];
      dest_index -= coord * dest_strides[i];
      src_index += src_strides[i] * (coord + offsets[i]);
  }
  return src_index;
}

template <typename Dtype>
__global__ void crop_kernel_forward(const int nthreads,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int src_index = compute_uncropped_index(
        index, ndims, src_strides, dest_strides, offsets);
    dest[index] = src[src_index];
  }
}

template <typename Dtype>
__global__ void crop_kernel_backward(const int nthreads,
    const int ndims,
    const int* src_strides,
    const int* dest_strides,
    const int* offsets,
    Dtype* src, const Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int src_index = compute_uncropped_index(
        index, ndims, src_strides, dest_strides, offsets);
    src[src_index] = dest[index];
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int n = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  crop_kernel_forward<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n,
      bottom[0]->num_axes(),
      src_strides_.gpu_data(),
      dest_strides_.gpu_data(),
      offsets.gpu_data(),
      bottom_data, top_data);
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int n = top[0]->count();

  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    // NOLINT_NEXT_LINE(whitespace/operators)
    crop_kernel_backward<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n,
        bottom[0]->num_axes(),
        src_strides_.gpu_data(),
        dest_strides_.gpu_data(),
        offsets.gpu_data(),
        bottom_diff, top_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
