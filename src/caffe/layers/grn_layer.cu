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

#include "thrust/device_vector.h"

#include "caffe/layers/grn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_mul(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    for (int c = 0; c < channels; ++c) {
      data[(n * channels + c) * spatial_dim + s] *= channel_sum[index];
    }
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int num, const int channels,
    const int spatial_dim, Dtype* data, const Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    for (int c = 0; c < channels; ++c) {
      data[(n * channels + c) * spatial_dim + s] /= channel_sum[index];
    }
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void GRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* square_data = square_.mutable_gpu_data();
  Dtype* norm_data = norm_.mutable_gpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  caffe_copy(bottom[0]->count(), bottom_data, square_data);

  // square
  caffe_gpu_powx<Dtype>(bottom[0]->count(), square_data,
                        Dtype(2.0), square_data);
  // sum cross channel
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, square_data,
      norm_data);
  // add bias to prevent zero division
  caffe_gpu_add_scalar<Dtype>(num * spatial_dim, bias_, norm_data);
  // square root
  caffe_gpu_powx<Dtype>(num * spatial_dim, norm_data, Dtype(0.5), norm_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_data,
      norm_data);
}

template <typename Dtype>
void GRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  caffe_copy(top[0]->count(), top_diff, bottom_diff);

  if (dummy_backward_) {
    return;
  }

  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* norm_data = norm_.mutable_gpu_data();
  Dtype* temp_dot_data = temp_dot_.mutable_gpu_data();
  Dtype* temp_data = square_.mutable_gpu_data();  // just reuse the square_
  int num = top[0]->num();
  int channels = top[0]->channels();
  int spatial_dim = top[0]->height() * top[0]->width();

  caffe_copy(top[0]->count(), bottom_data, temp_data);

  // b_diff = t_diff / norm - dot(t_diff, t_data) / (norm)^2 * bottom_data
  // temp_dot_data = dot(t_diff, t_data)
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, top_diff, top_data,
      temp_dot_data);
  // temp_dot_data /= (norm)^2
  caffe_gpu_div<Dtype>(num * spatial_dim, temp_dot_data,
                       norm_data, temp_dot_data);
  caffe_gpu_div<Dtype>(num * spatial_dim, temp_dot_data,
                       norm_data, temp_dot_data);
  // bottom_diff = top_diff, bottom_diff /= norm
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, bottom_diff,
      norm_data);
  // temp_data = bottom_data, temp_data *= temp_dot_data
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_mul<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, temp_data,
      temp_dot_data);
  // bottom_diff += -temp_data
  caffe_gpu_axpy<Dtype>(top[0]->count(), Dtype(-1.0), temp_data,
      bottom_diff);
}


INSTANTIATE_LAYER_GPU_FUNCS(GRNLayer);


}  // namespace caffe
