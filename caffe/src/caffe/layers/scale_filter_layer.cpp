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

#include "caffe/layers/scale_filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScaleFilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(bottom[0]->num_axes(), 4);

  min_scale_ = this->layer_param_.scale_filter_param().min_scale();
  max_scale_ = this->layer_param_.scale_filter_param().max_scale();
  DCHECK(min_scale_ > Dtype(0));
  DCHECK(max_scale_ > min_scale_);
}

template <typename Dtype>
void ScaleFilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation with
  // shape: [batch, num_channels]
  vector<int> shape(2, 1);
  shape[0] = bottom[0]->shape(0);
  shape[1] = bottom[0]->shape(1);
  rand_vec_.Reshape(shape);
}

template <typename Dtype>
void ScaleFilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  if (this->phase_ == TRAIN) {
    const int num = bottom[0]->shape(0);
    const int channels = bottom[0]->shape(1);
    const int dim = bottom[0]->shape(2) * bottom[0]->shape(3);
    Dtype* scales = rand_vec_.mutable_cpu_data();

    caffe_rng_uniform(num * channels, min_scale_, max_scale_, scales);

    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        caffe_scal(dim, scales[n * channels + c],
                   top_data + dim * (n * channels + c));
      }
    }
  }
}

template <typename Dtype>
void ScaleFilterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    caffe_copy(top[0]->count(), top_diff, bottom_diff);

    if (this->phase_ == TRAIN) {
      const int num = bottom[0]->shape(0);
      const int channels = bottom[0]->shape(1);
      const int dim = bottom[0]->shape(2) * bottom[0]->shape(3);
      const Dtype* scales = rand_vec_.cpu_data();

      for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
          caffe_scal(dim, scales[n * channels + c],
                     bottom_diff + dim * (n * channels + c));
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ScaleFilterLayer);
#endif

INSTANTIATE_CLASS(ScaleFilterLayer);
REGISTER_LAYER_CLASS(ScaleFilter);

}  // namespace caffe
