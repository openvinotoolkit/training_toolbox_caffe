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

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Slice(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_slice_size = slice_size * top_slice_axis;
    const int slice_num = index / total_slice_size;
    const int slice_index = index % total_slice_size;
    const int bottom_index = slice_index +
        (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) { return; }
  int offset_slice_axis = 0;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = true;
  for (int i = 0; i < top.size(); ++i) {
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
    Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data);
    offset_slice_axis += top_slice_axis;
  }
}

template <typename Dtype>
void SliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
  int offset_slice_axis = 0;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  const bool kForward = false;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    const int top_slice_size = top_slice_axis * slice_size_;
    const int nthreads = top_slice_size * num_slices_;
    Slice<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top_diff, kForward, num_slices_, slice_size_,
        bottom_slice_axis, top_slice_axis, offset_slice_axis, bottom_diff);
    offset_slice_axis += top_slice_axis;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SliceLayer);

}  // namespace caffe
