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

#ifndef CAFFE_MULTIBOX_LOSS_LAYER_HPP_
#define CAFFE_MULTIBOX_LOSS_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Perform MultiBox operations. Including the following:
 *
 *  - decode the predictions.
 *  - perform matching between priors/predictions and ground truth.
 *  - use matched boxes and confidences to compute loss.
 *
 *  Bottom blobs:
 *  - bottom[0] stores the location predictions
 *  - bottom[1] stores the confidence predictions
 *  - bottom[2] stores the prior bounding boxes
 *  - bottom[3] stores the ground truth bounding boxes
 */
template <typename Dtype>
class MultiBoxLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiBoxLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiBoxLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// The internal localization loss layer.
  shared_ptr<Layer<Dtype> > loc_loss_layer_;
  /// Type of normalization to get location loss
  LocLossType loc_loss_type_;
  /// Parameter to re-weight the location loss in total sum
  float loc_weight_;
  /// Bottom vector holder used in Forward function
  vector<Blob<Dtype>*> loc_bottom_vec_;
  /// Top vector holder used in Forward function
  vector<Blob<Dtype>*> loc_top_vec_;
  /// Stores the matched location prediction
  Blob<Dtype> loc_pred_;
  /// Stores the corresponding matched ground truth
  Blob<Dtype> loc_gt_;
  /// Stores localization loss
  Blob<Dtype> loc_loss_;
  /// The internal confidence loss layer.
  shared_ptr<Layer<Dtype> > conf_loss_layer_;
  /// Type of normalization to get confidence loss
  ConfLossType conf_loss_type_;
  /// Bottom vector holder used in Forward function
  vector<Blob<Dtype>*> conf_bottom_vec_;
  /// Top vector holder used in Forward function
  vector<Blob<Dtype>*> conf_top_vec_;
  /// Stores the confidence prediction
  Blob<Dtype> conf_pred_;
  /// Stores the corresponding ground truth label
  Blob<Dtype> conf_gt_;
  /// Stores confidence loss
  Blob<Dtype> conf_loss_;
  /// Stores parameters of layer
  MultiBoxLossParameter multibox_loss_param_;
  /// Number of classes to detect
  int num_classes_;
  /// Whether to share locations between differen classes
  bool share_location_;
  /// Type of bbox mathing strategu
  MatchType match_type_;
  /// Whether to use prior boxes instead of predicted to perform matching
  bool use_prior_for_matching_;
  /// Identificator of "background" class
  int background_label_id_;
  /// Whether to use gt bboxes marked as difficult for training
  bool use_difficult_gt_;
  /// Whether to perform mining procedure to fix positive samples imbalance problem
  bool do_neg_mining_;
  /// Type of mining procedure
  MiningType mining_type_;
  /// Number of location classes (depends on share_location_ parameter)
  int loc_classes_;
  /// Number of gt detections
  int num_gt_;
  /// Number of candidate detections
  int num_;
  /// Total number of prior boxes
  int num_priors_;
  /// Number of mathed with gt bbox
  int num_matches_;
  /// Number of samples in training (depends on whether mining is used)
  int num_conf_;
  /// Internal buffer to store matched detections
  vector<map<int, vector<int> > > all_match_indices_;
  /// Internal buffer to store negative detections
  vector<vector<int> > all_neg_indices_;
  /// Type of total loss normalization
  LossParameter_NormalizationMode normalization_;
};

}  // namespace caffe

#endif  // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
