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

#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
                           const Dtype* bottom_data_b, const int blob_idx,
                           Dtype* top_data, int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
    case EltwiseParameter_EltwiseOp_PROD:
      caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                    top_data);
      for (int i = 2; i < bottom.size(); ++i) {
        caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
      }
      break;
    case EltwiseParameter_EltwiseOp_SUM:
      //      caffe_gpu_set(count, Dtype(0.), top_data);
      if (bottom[0]->gpu_data() != top_data) {
        caffe_copy(count, bottom[0]->gpu_data(), top_data);
      }
      caffe_gpu_scal(count, coeffs_[0], top_data);

      // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
      for (int i = 1; i < bottom.size(); ++i) {
        assert(bottom[i]->gpu_data() != top_data);
        caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
      }
      break;
    case EltwiseParameter_EltwiseOp_MAX:
      mask = max_idx_.mutable_gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data,
          mask);
      for (int i = 2; i < bottom.size(); ++i) {
        // NOLINT_NEXT_LINE(whitespace/operators)
        MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_data, bottom[i]->gpu_data(), i - 1, top_data, mask);
      }
      break;
    default:
      LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
__global__ void MaxBackward(const int nthreads, const Dtype* top_diff,
                            const int blob_idx, const int* mask,
                            Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      switch (op_) {
        case EltwiseParameter_EltwiseOp_PROD:
          if (stable_prod_grad_) {
            bool initialized = false;
            for (int j = 0; j < bottom.size(); ++j) {
              if (i == j) {
                continue;
              }
              if (!initialized) {
                caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
                initialized = true;
              } else {
                caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
                              bottom_diff);
              }
            }
          } else {
            caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
          }
          caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
          break;
        case EltwiseParameter_EltwiseOp_SUM:
          if (coeffs_[i] == Dtype(1.)) {
            caffe_copy(count, top_diff, bottom_diff);
          } else {
            caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
          }
          break;
        case EltwiseParameter_EltwiseOp_MAX:
          mask = max_idx_.gpu_data();
          MaxBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
              <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                  count, top_diff, i, mask, bottom_diff);
          break;
        default:
          LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);

}  // namespace caffe
