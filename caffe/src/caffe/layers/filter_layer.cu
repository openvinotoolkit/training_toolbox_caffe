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

#include "caffe/layers/filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int new_tops_num = indices_to_forward_.size();
  // forward all filtered items for all bottoms but the Selector (bottom[last])
  for (int t = 0; t < top.size(); ++t) {
    const Dtype* bottom_data = bottom[t]->gpu_data();
    Dtype* top_data = top[t]->mutable_gpu_data();
    int dim = bottom[t]->count() / bottom[t]->shape(0);
    for (int n = 0; n < new_tops_num; ++n) {
      int data_offset_top = n * dim;
      int data_offset_bottom = indices_to_forward_[n] * dim;
      caffe_copy(dim, bottom_data + data_offset_bottom,
          top_data + data_offset_top);
    }
  }
}

template <typename Dtype>
void FilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[bottom.size() - 1]) {
    LOG(FATAL) << this->type()
               << "Layer cannot backpropagate to filter index inputs";
  }
  for (int i = 0; i < top.size(); ++i) {
    // bottom[last] is the selector and never needs backpropagation
    // so we can iterate over top vector because top.size() == bottom.size() -1
    if (propagate_down[i]) {
      const int dim = top[i]->count() / top[i]->shape(0);
      int next_to_backward_offset = 0;
      int batch_offset = 0;
      int data_offset_bottom = 0;
      int data_offset_top = 0;
      for (int n = 0; n < bottom[i]->shape(0); ++n) {
        if (next_to_backward_offset >= indices_to_forward_.size()) {
          // we already visited all items that were been forwarded, so
          // just set to zero remaining ones
          data_offset_bottom = n * dim;
          caffe_gpu_set(dim, Dtype(0),
              bottom[i]->mutable_gpu_diff() + data_offset_bottom);
        } else {
          batch_offset = indices_to_forward_[next_to_backward_offset];
          data_offset_bottom = n * dim;
          if (n != batch_offset) {  // this data was not been forwarded
            caffe_gpu_set(dim, Dtype(0),
                bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          } else {  // this data was been forwarded
            data_offset_top = next_to_backward_offset * dim;
            ++next_to_backward_offset;  // point to next forwarded item index
            caffe_copy(dim, top[i]->mutable_gpu_diff() + data_offset_top,
                bottom[i]->mutable_gpu_diff() + data_offset_bottom);
          }
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FilterLayer);

}  // namespace caffe
