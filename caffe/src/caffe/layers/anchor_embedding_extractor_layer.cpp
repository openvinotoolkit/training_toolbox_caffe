/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <map>
#include <vector>

#include "caffe/layers/anchor_embedding_extractor_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AnchorEmbeddingExtractorLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  AnchorEmbeddingExtractorParameter params =
      this->layer_param_.embedding_extractor_param();

  CHECK(params.has_num_anchors());
  num_anchors_ = params.num_anchors();
  CHECK_EQ(num_anchors_, 4) << "Support 4 anchors only";

  if (params.has_num_valid_actions()) {
    num_valid_actions_ = params.num_valid_actions();
  } else {
    num_valid_actions_ = -1;
  }

  CHECK(params.has_label_src_pos());
  label_src_pos_ = params.label_src_pos();
  CHECK_GE(label_src_pos_, 0);

  output_instance_weights_ = params.output_instance_weights();
  output_proposals_ = params.output_proposals();
  if (output_instance_weights_ && output_proposals_) {
    LOG(ERROR) << "Cannot output both variants simultaneously";
  }
  if (top.size() == 3 && !output_instance_weights_ && !output_proposals_) {
    LOG(ERROR) << "The output mode is not specified";
  }
}

template <typename Dtype>
void AnchorEmbeddingExtractorLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), num_anchors_ + 1);

  embedding_size_ = bottom[1]->channels();
  CHECK_GT(embedding_size_, 0);
  height_ = bottom[1]->height();
  CHECK_GT(height_, 0);
  width_ = bottom[1]->width();
  CHECK_GT(width_, 0);
  for (size_t i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[i]->channels(), embedding_size_)
        << "Number of channels in different anchors must be the same";
    CHECK_EQ(bottom[i]->height(), height_)
        << "Height in different anchors must be the same";
    CHECK_EQ(bottom[i]->width(), width_)
        << "Width in different anchors must be the same";
  }

  // fake shapes
  vector<int> shape(1);
  shape[0] = 1;
  top[1]->Reshape(shape);
  shape.push_back(embedding_size_);
  top[0]->Reshape(shape);

  valid_detection_ids_.reserve(num_anchors_ * height_ * width_);

  const bool do_instance_weighting =
      top.size() == 3 && output_instance_weights_;
  if (do_instance_weighting) {
    vector<int> out_shape(1);
    out_shape[0] = 1;
    top[2]->Reshape(out_shape);
  }

  const bool output_proposals = top.size() == 3 && output_proposals_;
  if (output_proposals) {
    vector<int> out_shape(2);
    out_shape[0] = 1;
    out_shape[1] = PROPOSAL_RECORD_SIZE;
    top[2]->Reshape(out_shape);
  }
}

template <typename Dtype>
void AnchorEmbeddingExtractorLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->width(), MATCHED_DET_RECORD_SIZE);

  const bool do_instance_weighting =
      top.size() == 3 && output_instance_weights_;
  const bool output_proposals =
      top.size() == 3 && output_proposals_;

  const Dtype* det_data = bottom[0]->cpu_data();
  const int num_detections = bottom[0]->height();
  if (do_instance_weighting) {
    GetValidDetectionIdsCPU(det_data, num_detections,
                            &valid_detection_ids_, &instance_weights_);
  } else {
    GetValidDetectionIdsCPU(det_data, num_detections,
                            &valid_detection_ids_, NULL);
  }

  if (valid_detection_ids_.empty()) {
    // fake shapes
    vector<int> out_shape(1);
    out_shape[0] = 1;
    top[1]->Reshape(out_shape);
    if (do_instance_weighting) {
      top[2]->Reshape(out_shape);
    }
    out_shape.push_back(embedding_size_);
    top[0]->Reshape(out_shape);
    if (output_proposals) {
      out_shape[1] = PROPOSAL_RECORD_SIZE;
      top[2]->Reshape(out_shape);
    }

    caffe_set(MATCHED_DET_RECORD_SIZE, Dtype(0), top[0]->mutable_cpu_data());
    caffe_set(1, Dtype(-1), top[1]->mutable_cpu_data());
    if (do_instance_weighting) {
      caffe_set(1, Dtype(-1), top[2]->mutable_cpu_data());
    }
    if (output_proposals) {
      caffe_set(PROPOSAL_RECORD_SIZE, Dtype(-1), top[2]->mutable_cpu_data());
    }
  } else {
    vector<int> out_shape(1);
    out_shape[0] = static_cast<int>(valid_detection_ids_.size());
    top[1]->Reshape(out_shape);
    if (do_instance_weighting) {
      top[2]->Reshape(out_shape);
    }
    out_shape.push_back(embedding_size_);
    top[0]->Reshape(out_shape);
    if (output_proposals) {
      out_shape[1] = PROPOSAL_RECORD_SIZE;
      top[2]->Reshape(out_shape);
    }

    vector<const Dtype*> anchors;
    for (size_t i = 1; i < bottom.size(); ++i) {
      anchors.push_back(bottom[i]->cpu_data());
    }

    Dtype* proposals_data = NULL;
    if (output_proposals) {
      proposals_data = top[2]->mutable_cpu_data();
    }

    Dtype* embeddings_data = top[0]->mutable_cpu_data();
    Dtype* labels_data = top[1]->mutable_cpu_data();
    for (size_t det_id = 0; det_id < valid_detection_ids_.size(); ++det_id) {
      const int detection_start =
          static_cast<int>(valid_detection_ids_[det_id]) *
                           MATCHED_DET_RECORD_SIZE;
      const int item_id =
          static_cast<int>(det_data[detection_start + ITEM_ID_POS]);
      const int anchor =
          static_cast<int>(det_data[detection_start + ANCHOR_POS]);
      const int x_pos =
          static_cast<int>(det_data[detection_start + X_POS]);
      const int y_pos =
          static_cast<int>(det_data[detection_start + Y_POS]);
      const int label =
          static_cast<Dtype>(det_data[detection_start + label_src_pos_]);

      labels_data[det_id] = label;

      const int anchor_start = item_id * embedding_size_ * height_ * width_;
      for (int i = 0; i < embedding_size_; ++i) {
        const int shift =
            anchor_start + i * height_ * width_ + y_pos * width_ + x_pos;
        embeddings_data[det_id * embedding_size_ + i] = anchors[anchor][shift];
      }

      if (output_proposals) {
        proposals_data[det_id * PROPOSAL_RECORD_SIZE] =
            item_id;
        proposals_data[det_id * PROPOSAL_RECORD_SIZE + 1] =
            det_data[detection_start + X_MIN_POS];
        proposals_data[det_id * PROPOSAL_RECORD_SIZE + 2] =
            det_data[detection_start + Y_MIN_POS];
        proposals_data[det_id * PROPOSAL_RECORD_SIZE + 3] =
            det_data[detection_start + X_MAX_POS];
        proposals_data[det_id * PROPOSAL_RECORD_SIZE + 4] =
            det_data[detection_start + Y_MAX_POS];
      }
    }

    if (do_instance_weighting) {
      caffe_copy(instance_weights_.size(),
                 &instance_weights_.front(),
                 top[2]->mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void AnchorEmbeddingExtractorLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to matched detections.";
  }

  vector<Dtype*> anchor_diff;
  for (size_t i = 1; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* diff = bottom[i]->mutable_cpu_diff();
      caffe_set(bottom[i]->count(), Dtype(0), diff);

      anchor_diff.push_back(diff);
    } else {
      anchor_diff.push_back(NULL);
    }
  }

  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (size_t det_id = 0; det_id < valid_detection_ids_.size(); ++det_id) {
    const int detection_start =
        static_cast<int>(valid_detection_ids_[det_id]) *
                         MATCHED_DET_RECORD_SIZE;
    const int anchor =
        static_cast<int>(det_data[detection_start + ANCHOR_POS]);
    if (propagate_down[anchor + 1]) {
      const int item_id =
          static_cast<int>(det_data[detection_start + ITEM_ID_POS]);
      const int x_pos =
          static_cast<int>(det_data[detection_start + X_POS]);
      const int y_pos =
          static_cast<int>(det_data[detection_start + Y_POS]);
      const int anchor_start =
          item_id * embedding_size_ * height_ * width_;

      for (int i = 0; i < embedding_size_; ++i) {
        const int shift =
            anchor_start + i * height_ * width_ + y_pos * width_ + x_pos;
        anchor_diff[anchor][shift] += top_diff[det_id * embedding_size_ + i];
      }
    }
  }
}

template <typename Dtype>
void AnchorEmbeddingExtractorLayer<Dtype>::GetValidDetectionIdsCPU(
    const Dtype* data, int num_records,
    vector<Dtype>* ids, vector<Dtype>* weights) {
  const bool do_instance_counting = weights != NULL;

  ids->clear();

  map<int, int> instances;
  for (int i = 0; i < num_records; ++i) {
    const int item_id =
        static_cast<int>(data[i * MATCHED_DET_RECORD_SIZE + ITEM_ID_POS]);
    if (item_id < 0) {
      continue;
    }

    const int action =
        static_cast<int>(data[i * MATCHED_DET_RECORD_SIZE + ACTION_POS]);
    if (num_valid_actions_ < 0 || action < num_valid_actions_) {
      ids->push_back(Dtype(i));

      if (do_instance_counting) {
        const int id =
            static_cast<int>(data[i * MATCHED_DET_RECORD_SIZE + TRACK_ID_POS]);
        const int instance_id =
            id * 1000 + item_id;

        if (instances.find(instance_id) != instances.end()) {
          instances[instance_id] += 1;
        } else {
          instances[instance_id] = 1;
        }
      }
    }
  }

  if (do_instance_counting) {
    weights->resize(ids->size());

    for (size_t i = 0; i < ids->size(); ++i) {
      const int record_id =
          static_cast<int>((*ids)[i]);
      const int item_id =
          static_cast<int>(data[record_id * MATCHED_DET_RECORD_SIZE +
                                ITEM_ID_POS]);
      const int id =
          static_cast<int>(data[record_id * MATCHED_DET_RECORD_SIZE +
                                TRACK_ID_POS]);
      const int instance_id =
          id * 1000 + item_id;

      (*weights)[i] = Dtype(1) / Dtype(instances[instance_id]);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AnchorEmbeddingExtractorLayer);
#endif

INSTANTIATE_CLASS(AnchorEmbeddingExtractorLayer);
REGISTER_LAYER_CLASS(AnchorEmbeddingExtractor);

}  // namespace caffe
