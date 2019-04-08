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

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.loss_weight_size() == 0 && top.size() == 1) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }

  LayerParameter softmax_param;
  softmax_param.set_name(this->layer_param_.name() + "_internal_softmax");
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  max_entropy_weight_ = this->layer_param_.loss_param().max_entropy_weight();
  add_max_entropy_term_ = max_entropy_weight_ > Dtype(0);

  do_instance_weighting_ = bottom.size() == 3;

  weight_by_label_freqs_ =
    this->layer_param_.loss_param().weight_by_label_freqs();
  adaptive_weighting_ =
    this->layer_param_.loss_param().adaptive_weighting();
  if (weight_by_label_freqs_ && adaptive_weighting_) {
    LOG(ERROR) << "Cannot use both weighting schemes simultaneously";
  }

  use_weighting_ = weight_by_label_freqs_ || adaptive_weighting_;
  if (use_weighting_) {
    const int num_classes = bottom[0]->shape(softmax_axis_);
    CHECK_GT(num_classes, 0) << "Number of classes should be greater than 0";

    vector<int> weights_shape(1, num_classes);
    class_weights_.Reshape(weights_shape);

    if (weight_by_label_freqs_) {
      CHECK_EQ(this->layer_param_.loss_param().class_weighting_size(),
               num_classes)
          << "Number of class weight values does not match"
             "the number of classes.";

      Dtype* class_weights_data = class_weights_.mutable_cpu_data();
      for (int i = 0; i < num_classes; ++i) {
          class_weights_data[i] =
              this->layer_param_.loss_param().class_weighting(i);
      }
    } else if (adaptive_weighting_) {
      class_counts_.Reshape(weights_shape);
      smoothed_frequencies_.Reshape(weights_shape);
      caffe_set(num_classes, Dtype(-1),
                smoothed_frequencies_.mutable_cpu_data());

      gamma_ = this->layer_param_.loss_param().gamma();
      CHECK_GT(gamma_, Dtype(0));
      CHECK_LE(gamma_, Dtype(1));

      scale_ = this->layer_param_.loss_param().scale();
      CHECK_GT(scale_, Dtype(0));

      has_min_weight_ = this->layer_param_.loss_param().has_min_weight();
      if (has_min_weight_) {
        min_weight_ = this->layer_param_.loss_param().min_weight();
        CHECK_GE(min_weight_, Dtype(0));
      }

      has_max_weight_ = this->layer_param_.loss_param().has_max_weight();
      if (has_max_weight_) {
        max_weight_ = this->layer_param_.loss_param().max_weight();
        CHECK_GE(max_weight_, Dtype(0));

        if (has_min_weight_) {
          CHECK_GT(max_weight_, min_weight_);
        }
      }
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

  vector<int> loss_bufer_shape(2);
  loss_bufer_shape[0] = outer_num_;
  loss_bufer_shape[1] = inner_num_;
  valid_losses_.Reshape(loss_bufer_shape);

  const int num_top_blobs = top.size();
  if (num_top_blobs > 1) {
    if (num_top_blobs == 2) {
      top[1]->ReshapeLike(*bottom[0]);
    } else {
      const int num_classes = bottom[0]->shape(softmax_axis_);
      CHECK_EQ(num_top_blobs, 2 * num_classes + 1);

      const vector<int> loss_shape(0);
      for (int i = 1; i < 2 * num_classes + 1; ++i) {
        top[i]->Reshape(loss_shape);
      }
    }
  }

  if (weight_by_label_freqs_) {
    CHECK_EQ(class_weights_.count(), bottom[0]->shape(softmax_axis_))
        << "Number of input classes does not match the number of classes.";
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order
  // to 'turn off' a particular loss in a multi-task setup.
  // The max prevents NaNs in that case.
  return std::max(Dtype(1), normalizer);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::EstimateClassWeights_cpu(
    int num_classes, const Dtype* estimated_values,
    Dtype* smoothed_values, Dtype* weights) {
  Dtype sum_smoothed_losses = 0;
  for (int i = 0; i < num_classes; ++i) {
    Dtype value = estimated_values[i];

    if (smoothed_values[i] > Dtype(0)) {
      smoothed_values[i] =
          gamma_ * smoothed_values[i] + (Dtype(1) - gamma_) * value;
    } else {
      smoothed_values[i] = value;
    }

    if (smoothed_values[i] > Dtype(0)) {
      sum_smoothed_losses += smoothed_values[i];
    }
  }

  Dtype norm_factor = sum_smoothed_losses / Dtype(num_classes);
  for (int i = 0; i < num_classes; ++i) {
    Dtype weight =
        smoothed_values[i] > Dtype(0)
           ? norm_factor / smoothed_values[i]
           : Dtype(0);

    if (has_min_weight_ && weight < min_weight_) {
      weight = min_weight_;
    }
    if (has_max_weight_ && weight > max_weight_) {
      weight = max_weight_;
    }

    weights[i] = scale_ * weight;
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  const int dim = prob_.count() / outer_num_;
  const int num_classes = bottom[0]->shape(softmax_axis_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_cpu_diff();
  caffe_set(outer_num_ * inner_num_, Dtype(0), loss_data);

  Dtype* valid_loss_data = valid_losses_.mutable_cpu_data();
  caffe_set(outer_num_ * inner_num_, Dtype(0), valid_loss_data);

  Dtype* class_counts_data = NULL;
  if (adaptive_weighting_) {
    class_counts_data = class_counts_.mutable_cpu_data();
    caffe_set(num_classes, Dtype(0), class_counts_data);
  }

  const Dtype* instance_weights_data = NULL;
  if (do_instance_weighting_) {
    instance_weights_data = bottom[2]->cpu_data();
  }

  int num_valid = 0;
  Dtype num_instances = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }

      CHECK_GE(label_value, 0);
      CHECK_LT(label_value, num_classes);

      const int idx = i * dim + label_value * inner_num_ + j;
      Dtype loss_value = -log(std::max(prob_data[idx], Dtype(FLT_MIN)));

      if (add_max_entropy_term_) {
        Dtype neg_entropy = 0;
        for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
          const int prob_idx = i * dim + c * inner_num_ + j;
          neg_entropy +=
              log(std::max(prob_data[prob_idx],
                           Dtype(FLT_MIN))) * prob_data[prob_idx];
        }

        loss_value += max_entropy_weight_ * neg_entropy;
      }

      if (loss_value > Dtype(0)) {
        if (do_instance_weighting_) {
          loss_value *= instance_weights_data[i * inner_num_ + j];
          num_instances += instance_weights_data[i * inner_num_ + j];
        }

        loss_data[i * inner_num_ + j] = loss_value;
        valid_loss_data[i * inner_num_ + j] = Dtype(1);

        if (adaptive_weighting_) {
          class_counts_data[label_value] += Dtype(1);
        }

        ++num_valid;
      }
    }
  }

  if (adaptive_weighting_) {
    Dtype factor = num_valid > 0 ? Dtype(1) / Dtype(num_valid) : Dtype(0);
    for (int i = 0; i < num_classes; ++i) {
      class_counts_data[i] *= factor;
    }

    EstimateClassWeights_cpu(num_classes, class_counts_.cpu_data(),
                             smoothed_frequencies_.mutable_cpu_data(),
                             class_weights_.mutable_cpu_data());
  }

  const Dtype* class_weights_data = NULL;
  if (use_weighting_) {
    class_weights_data = class_weights_.cpu_data();
  }

  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int idx = i * inner_num_ + j;
      if (valid_loss_data[idx] > Dtype(0)) {
        const int label_value = static_cast<int>(label[idx]);
        if (use_weighting_) {
          loss += loss_data[idx] * class_weights_data[label_value];
        } else {
          loss += loss_data[idx];
        }
      }
    }
  }

  if (do_instance_weighting_) {
    normalizer_ =
        Dtype(1) / get_normalizer(normalization_,
                                  static_cast<int>(num_instances));
  } else {
    normalizer_ =
        Dtype(1) / get_normalizer(normalization_, num_valid);
  }
  top[0]->mutable_cpu_data()[0] = loss * normalizer_;

  if (top.size() > 1) {
    if (top.size() == 2) {
      top[1]->ShareData(prob_);
    } else {
      CHECK(adaptive_weighting_);

      const Dtype* smoothed_values_data = smoothed_frequencies_.cpu_data();
      for (int i = 0; i < num_classes; ++i) {
        top[2 * i + 1]->mutable_cpu_data()[0] = smoothed_values_data[i];
        top[2 * i + 2]->mutable_cpu_data()[0] = class_weights_data[i];
      }
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (top.size() == 3 && propagate_down[2]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to instance weights.";
  }

  if (propagate_down[0]) {
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* valid_losses = valid_losses_.cpu_data();
    const Dtype* class_weights_data = NULL;
    if (use_weighting_) {
      class_weights_data = class_weights_.cpu_data();
    }
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    caffe_copy(prob_.count(), prob_data, bottom_diff);

    const Dtype* instance_weights_data = NULL;
    if (do_instance_weighting_) {
      instance_weights_data = bottom[2]->cpu_data();
    }

    int dim = prob_.count() / outer_num_;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int plain_idx = i * inner_num_ + j;
        if (valid_losses[plain_idx] > Dtype(0)) {
          if (add_max_entropy_term_) {
            Dtype entropy = 0;
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              const int prob_idx = i * dim + c * inner_num_ + j;
              entropy +=
                  -log(std::max(prob_data[prob_idx],
                                Dtype(FLT_MIN))) * prob_data[prob_idx];
            }

            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              const int prob_idx = i * dim + c * inner_num_ + j;
              bottom_diff[prob_idx] *= Dtype(1) + max_entropy_weight_ *
                  (entropy + log(std::max(prob_data[prob_idx],
                                          Dtype(FLT_MIN))));
            }
          }

          const int label_value = static_cast<int>(label[plain_idx]);
          const int idx = i * dim + label_value * inner_num_ + j;
          bottom_diff[idx] -= 1;

          if (use_weighting_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] *=
                  class_weights_data[label_value];
            }
          }

          if (do_instance_weighting_) {
            for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              bottom_diff[i * dim + c * inner_num_ + j] *=
                  instance_weights_data[plain_idx];
            }
          }
        } else {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = Dtype(0);
          }
        }
      }
    }

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] * normalizer_;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
