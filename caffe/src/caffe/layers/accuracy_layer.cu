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

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AccuracyForwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* label, Dtype* acc,
          const int num, const int dim, const int spatial_dim,
          const int num_labels, const int top_k,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const Dtype prob_of_true_class = bottom_data[n * dim
                                                 + label_value * spatial_dim
                                                 + s];
    int num_better_predictions = -1;  // true_class also counts as "better"
    if (has_ignore_label_ && label_value == ignore_label_) {
      acc[index] = 0;
      counts[index] = 0;
    } else {
      for (int k = 0; k < num_labels & num_better_predictions < top_k; k++) {
        num_better_predictions +=
          (bottom_data[n * dim + k * spatial_dim + s] >= prob_of_true_class);
      }
      acc[index] = (num_better_predictions < top_k);
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void AccuracyForwardWithPerClassGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* label,
          Dtype* acc, Dtype* counts,
          const int num, const int dim, const int spatial_dim,
          const int num_labels, const int top_k,
          const bool has_ignore_label_, const int ignore_label_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const Dtype prob_of_true_class = bottom_data[n * dim
                                                 + label_value * spatial_dim
                                                 + s];
    if (has_ignore_label_ && label_value == ignore_label_) {
      // nothing to be done.
    } else {
      int num_better_predictions = -1;  // true_class also counts as "better"
      for (int k = 0; k < num_labels & num_better_predictions < top_k; k++) {
        num_better_predictions +=
          (bottom_data[n * dim + k * spatial_dim + s] >= prob_of_true_class);
      }
      acc[label_value*nthreads + index] += (num_better_predictions < top_k);
      counts[label_value*nthreads + index] = 1;
    }
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_label = bottom[1]->gpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  Dtype* acc_data = bottom[0]->mutable_gpu_diff();
  if (top.size() == 1) {
    // simple case - report only global accuracy.

    // Similarly, this memory is never used elsewhere, and thus we can use it
    // to avoid having to allocate additional GPU memory.
    Dtype* counts = bottom[1]->mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    AccuracyForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, bottom_label,
        acc_data, outer_num_, dim, inner_num_, num_labels, top_k_,
        has_ignore_label_, ignore_label_, counts);
    Dtype acc;
    caffe_gpu_asum(nthreads, acc_data, &acc);
    Dtype valid_count;
    caffe_gpu_asum(nthreads, counts, &valid_count);
    if (valid_count > 0) {
      top[0]->mutable_cpu_data()[0] = acc / valid_count;
    } else {
      top[0]->mutable_cpu_data()[0] = 0;
    }
  } else {
    // need to report per-class accuracy as well

    // allocate space for more detailed "counts"
    nums_buffer_.ReshapeLike(*bottom[0]);
    Dtype* counts = nums_buffer_.mutable_gpu_data();

    caffe_gpu_set(bottom[0]->count(), Dtype(0), acc_data);
    caffe_gpu_set(nums_buffer_.count(), Dtype(0), counts);

    // NOLINT_NEXT_LINE(whitespace/operators)
    AccuracyForwardWithPerClassGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, bottom_label,
        acc_data, counts, outer_num_, dim, inner_num_, num_labels, top_k_,
        has_ignore_label_, ignore_label_);

    // get the overall accuracy
    Dtype acc;
    caffe_gpu_asum(bottom[0]->count(), acc_data, &acc);
    Dtype valid_count;
    caffe_gpu_asum(nums_buffer_.count(), counts, &valid_count);
    if (valid_count > 0) {
      top[0]->mutable_cpu_data()[0] = acc / valid_count;
    } else {
      top[0]->mutable_cpu_data()[0] = 0;
    }

    // get per-class accuracy
    Dtype* per_class_acc = top[1]->mutable_cpu_data();
    for (int l = 0; l < num_labels; l++) {
      caffe_gpu_asum(nthreads, acc_data + l*nthreads, per_class_acc+l);
      caffe_gpu_asum(nthreads, counts + l*nthreads, &valid_count);
      if (valid_count > 0) {
        per_class_acc[l] /= valid_count;
      } else {
        per_class_acc[l] = 0;
      }
    }
  }
  // Clear scratch memory to prevent interfering with backward (see #6202).
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
}


template <typename Dtype>
void AccuracyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {  NOT_IMPLEMENTED;  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AccuracyLayer);
}  // namespace caffe
