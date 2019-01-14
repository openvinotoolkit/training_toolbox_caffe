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

#include <vector>

#include "caffe/layers/anchor_embedding_extractor_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Forward(const Dtype* det_data, const Dtype* ids, const int n,
                        const Dtype* embed1, const Dtype* embed2,
                        const Dtype* embed3, const Dtype* embed4,
                        int embedding_size, int height, int width,
                        int label_src_pos, Dtype* embeddings_data,
                        Dtype* labels_data, bool output_proposals,
                        Dtype* proposals_data) {
  CUDA_KERNEL_LOOP(index, n) {
    const int detection_start =
        static_cast<int>(ids[index]) * MATCHED_DET_RECORD_SIZE;
    const int item_id =
        static_cast<int>(det_data[detection_start + ITEM_ID_POS]);
    const int anchor =
        static_cast<int>(det_data[detection_start + ANCHOR_POS]);
    const int x_pos =
        static_cast<int>(det_data[detection_start + X_POS]);
    const int y_pos =
        static_cast<int>(det_data[detection_start + Y_POS]);
    const int label =
        static_cast<Dtype>(det_data[detection_start + label_src_pos]);

    labels_data[index] = label;

    const Dtype* embed_source = NULL;
    switch (anchor) {
      case 0: embed_source = embed1; break;
      case 1: embed_source = embed2; break;
      case 2: embed_source = embed3; break;
      case 3: embed_source = embed4; break;
    }

    const int anchor_start = item_id * embedding_size * height * width;
    for (int i = 0; i < embedding_size; ++i) {
      const int shift =
          anchor_start + i * height * width + y_pos * width + x_pos;
      embeddings_data[index * embedding_size + i] = embed_source[shift];
    }

    if (output_proposals) {
      proposals_data[index * PROPOSAL_RECORD_SIZE] =
          item_id;
      proposals_data[index * PROPOSAL_RECORD_SIZE + 1] =
          det_data[detection_start + X_MIN_POS];
      proposals_data[index * PROPOSAL_RECORD_SIZE + 2] =
          det_data[detection_start + Y_MIN_POS];
      proposals_data[index * PROPOSAL_RECORD_SIZE + 3] =
          det_data[detection_start + X_MAX_POS];
      proposals_data[index * PROPOSAL_RECORD_SIZE + 4] =
          det_data[detection_start + Y_MAX_POS];
    }
  }
}

template <typename Dtype>
__global__ void Backward(const Dtype* det_data, const Dtype* ids,
                         const int n, const Dtype* top_diff,
                         int embedding_size, int height, int width,
                         Dtype* embed1_diff, Dtype* embed2_diff,
                         Dtype* embed3_diff, Dtype* embed4_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const int detection_start =
        static_cast<int>(ids[index]) * MATCHED_DET_RECORD_SIZE;
    const int anchor =
        static_cast<int>(det_data[detection_start + ANCHOR_POS]);

    Dtype* embed_diff_target = NULL;
    switch (anchor) {
      case 0: embed_diff_target = embed1_diff; break;
      case 1: embed_diff_target = embed2_diff; break;
      case 2: embed_diff_target = embed3_diff; break;
      case 3: embed_diff_target = embed4_diff; break;
    }

    if (embed_diff_target != NULL) {
      const int item_id =
          static_cast<int>(det_data[detection_start + ITEM_ID_POS]);
      const int x_pos =
          static_cast<int>(det_data[detection_start + X_POS]);
      const int y_pos =
          static_cast<int>(det_data[detection_start + Y_POS]);
      const int anchor_start =
          item_id * embedding_size * height * width;

      for (int i = 0; i < embedding_size; ++i) {
        const int shift =
            anchor_start + i * height * width + y_pos * width + x_pos;
        embed_diff_target[shift] += top_diff[index * embedding_size + i];
      }
    }
  }
}

template <typename Dtype>
void AnchorEmbeddingExtractorLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->width(), MATCHED_DET_RECORD_SIZE);

  const bool do_instance_weighting =
      top.size() == 3 && output_instance_weights_;
  const bool output_proposals =
      top.size() == 3 && output_proposals_;

  const int num_detections = bottom[0]->height();
  if (do_instance_weighting) {
    GetValidDetectionIdsCPU(bottom[0]->cpu_data(), num_detections,
                            &valid_detection_ids_, &instance_weights_);
  } else {
    GetValidDetectionIdsCPU(bottom[0]->cpu_data(), num_detections,
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

    caffe_gpu_set(MATCHED_DET_RECORD_SIZE, Dtype(0),
                  top[0]->mutable_gpu_data());
    caffe_gpu_set(1, Dtype(-1),
                  top[1]->mutable_gpu_data());
    if (do_instance_weighting) {
      caffe_gpu_set(1, Dtype(-1),
                    top[2]->mutable_gpu_data());
    }
    if (output_proposals) {
      caffe_gpu_set(PROPOSAL_RECORD_SIZE, Dtype(-1),
                    top[2]->mutable_gpu_data());
    }
  } else {
    vector<int> out_shape(1);
    out_shape[0] = static_cast<int>(valid_detection_ids_.size());
    valid_ids_.Reshape(out_shape);
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

    caffe_copy(valid_detection_ids_.size(),
               &valid_detection_ids_.front(),
               valid_ids_.mutable_cpu_data());
    if (do_instance_weighting) {
      caffe_copy(instance_weights_.size(),
                 &instance_weights_.front(),
                 top[2]->mutable_gpu_data());
    }

    Dtype* proposals_data = NULL;
    if (output_proposals) {
      proposals_data = top[2]->mutable_gpu_data();
    }

    const Dtype* det_data = bottom[0]->gpu_data();
    Dtype* embeddings_data = top[0]->mutable_gpu_data();
    Dtype* labels_data = top[1]->mutable_gpu_data();

    const int count = valid_detection_ids_.size();
    Forward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        det_data, valid_ids_.gpu_data(), count,
        bottom[1]->gpu_data(), bottom[2]->gpu_data(),
        bottom[3]->gpu_data(), bottom[4]->gpu_data(),
        embedding_size_, height_, width_, label_src_pos_,
        embeddings_data, labels_data,
        output_proposals, proposals_data);
  }
}

template <typename Dtype>
void AnchorEmbeddingExtractorLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to matched detections.";
  }

  vector<Dtype*> anchor_diff;
  for (size_t i = 1; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* diff = bottom[i]->mutable_gpu_diff();
      caffe_gpu_set(bottom[i]->count(), Dtype(0), diff);

      anchor_diff.push_back(diff);
    } else {
      anchor_diff.push_back(NULL);
    }
  }

  const Dtype* det_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = valid_detection_ids_.size();
  if (count > 0) {
    Backward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      det_data, valid_ids_.gpu_data(), count, top_diff,
      embedding_size_, height_, width_,
      anchor_diff[0], anchor_diff[1], anchor_diff[2], anchor_diff[3]);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AnchorEmbeddingExtractorLayer);

}  // namespace caffe
