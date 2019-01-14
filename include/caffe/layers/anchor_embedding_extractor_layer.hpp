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

#ifndef CAFFE_ANCHOR_EMBEDDING_EXTRACTOR_LAYER_HPP_
#define CAFFE_ANCHOR_EMBEDDING_EXTRACTOR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#define MATCHED_DET_RECORD_SIZE 11
#define ITEM_ID_POS 0
#define ANCHOR_POS 6
#define ACTION_POS 8
#define X_POS 9
#define Y_POS 10
#define TRACK_ID_POS 7
#define X_MIN_POS 2
#define Y_MIN_POS 3
#define X_MAX_POS 4
#define Y_MAX_POS 5
#define PROPOSAL_RECORD_SIZE 5

namespace caffe {

/**
 * @brief Extracts embeddings from the specified SSD-based anchor branches
 *        according to the prediction info (anchor_id and pixel location).
 *
 */
template <typename Dtype>
class AnchorEmbeddingExtractorLayer : public Layer<Dtype> {
 public:
  explicit AnchorEmbeddingExtractorLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "AnchorEmbeddingExtractor"; }
  virtual inline int MinNumBottomBlobs() const { return 2; }
  virtual inline int MinNumTopBlobs() const { return 2; }
  virtual inline int MaxNumTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief Filter input detections to exclude low-confedence or
   *        invalid detections.
   *
   */
  void GetValidDetectionIdsCPU(const Dtype* data, int num_records,
                               vector<Dtype>* ids, vector<Dtype>* weights);

  /// Number of anchors in SSD head
  int num_anchors_;
  /// Number of classes to skip invalid detections from annotation
  int num_valid_actions_;
  /// Position to extract the class info from the input annotation
  int label_src_pos_;
  /// Size of embedding vector
  int embedding_size_;
  /// Height of input feature map to map the pixel position
  int height_;
  /// Width of input feature map to map the pixel position
  int width_;
  /// Whether to print current weights if size of layer top is enough
  bool output_instance_weights_;
  /// Whether to print proposals if size of layer top is enough
  bool output_proposals_;
  /// Stores positions of valid detections from the input annoation
  vector<Dtype> valid_detection_ids_;
  /// Stores number of appearances for each instance.
  vector<Dtype> instance_weights_;
  /// Buffer to work on the gpu device.
  Blob<Dtype> valid_ids_;
};

}  // namespace caffe

#endif  // CAFFE_ANCHOR_EMBEDDING_EXTRACTOR_LAYER_HPP_
