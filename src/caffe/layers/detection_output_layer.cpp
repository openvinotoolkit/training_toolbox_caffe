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
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const DetectionOutputParameter& detection_output_param =
      this->layer_param_.detection_output_param();

  CHECK(detection_output_param.has_num_classes())
      << "Must specify main num_classes";
  num_classes_ = detection_output_param.num_classes();
  CHECK_EQ(num_classes_, 2)
      << "Number of main classes must be 2";

  share_location_ = detection_output_param.share_location();
  num_loc_classes_ = share_location_ ? 1 : num_classes_;
  code_type_ = detection_output_param.code_type();
  variance_encoded_in_target_ =
      detection_output_param.variance_encoded_in_target();
  keep_top_k_ = detection_output_param.keep_top_k();
  confidence_threshold_ = detection_output_param.has_confidence_threshold() ?
      detection_output_param.confidence_threshold() : -FLT_MAX;

  // Parameters used in nms.
  nms_threshold_ = detection_output_param.nms_param().nms_threshold();
  CHECK_GE(nms_threshold_, 0.)
      << "nms_threshold must be non negative.";
  eta_ = detection_output_param.nms_param().eta();
  CHECK_GT(eta_, 0.);
  CHECK_LE(eta_, 1.);
  top_k_ = -1;
  if (detection_output_param.nms_param().has_top_k()) {
    top_k_ = detection_output_param.nms_param().top_k();
  }
}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int record_size = 10;

  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  num_priors_ = bottom[2]->height() / 4;
  CHECK_EQ(num_priors_ * num_loc_classes_ * 4, bottom[0]->count(1))
      << "Number of priors must match number of location predictions.";
  CHECK_EQ(num_priors_ * num_classes_, bottom[1]->count(1))
      << "Number of priors must match number of confidence predictions.";
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // Each row is a record_size dimension vector, which stores
  // [image_id, label, confidence, xmin, ymin, xmax, ymax]
  top_shape.push_back(record_size);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int record_size = 10;
  const Dtype* loc_data = bottom[0]->cpu_data();
  const Dtype* conf_data = bottom[1]->cpu_data();
  const Dtype* prior_data = bottom[2]->cpu_data();
  const int num = bottom[0]->num();
  const int height = bottom[0]->shape(1);
  const int width = bottom[0]->shape(2);
  const int num_anchors = bottom[0]->shape(3) / 4;

  // Retrieve all location predictions.
  vector<LabelBBox> all_loc_preds;
  GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                    share_location_, &all_loc_preds);

  // Retrieve all confidences.
  vector<map<int, vector<BBoxMatch<Dtype> > > > all_conf_scores;
  GetConfidenceScores(conf_data, num, height, width, num_anchors, num_classes_,
                      &all_conf_scores);

  // Retrieve all prior bboxes. It is same within a batch since we assume all
  // images in a batch are of same dimension.
  vector<NormalizedBBox> prior_bboxes;
  vector<vector<float> > prior_variances;
  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

  // Decode all loc predictions to bboxes.
  vector<LabelBBox> all_decode_bboxes;
  const bool clip_bbox = false;
  DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                  share_location_, num_loc_classes_, 0,
                  code_type_, variance_encoded_in_target_, clip_bbox,
                  &all_decode_bboxes);

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    const map<int, vector<BBoxMatch<Dtype> > >& conf_scores =
        all_conf_scores[i];

    map<int, vector<int> > indices;
    int num_det = 0;
    for (int c = 1; c < num_classes_; ++c) {
      if (conf_scores.find(c) == conf_scores.end()) {
        // Something bad happened if there are no predictions for label.
        LOG(FATAL)
            << "Could not find confidence predictions for label " << c;
      }

      const vector<BBoxMatch<Dtype> >& paired_scores =
          conf_scores.find(c)->second;
      vector<float> scores(paired_scores.size());
      for (size_t score_id = 0; score_id < paired_scores.size(); ++score_id) {
        scores[score_id] = paired_scores[score_id].conf;
      }

      int label = share_location_ ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for label.
        LOG(FATAL)
            << "Could not find location predictions for label " << label;
        continue;
      }
      const vector<NormalizedBBox>& bboxes =
          decode_bboxes.find(label)->second;
      ApplyNMSFast(bboxes, scores, confidence_threshold_,
                   nms_threshold_, eta_, top_k_, &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        if (conf_scores.find(label) == conf_scores.end()) {
          // Something bad happened for current label.
          LOG(FATAL)
              << "Could not find location predictions for " << label;
          continue;
        }
        const vector<BBoxMatch<Dtype> >& scores =
            conf_scores.find(label)->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          CHECK_LT(idx, scores.size());
          score_index_pairs.push_back(std::make_pair(
                  scores[idx].conf, std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(record_size);
  Dtype* top_data;
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    top_shape[2] = num;
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      top_data += record_size;
    }
  } else {
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
  }

  int count = 0;
  for (int i = 0; i < num; ++i) {
    const map<int, vector<BBoxMatch<Dtype> > >& conf_scores =
        all_conf_scores[i];
    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      if (conf_scores.find(label) == conf_scores.end()) {
        // Something bad happened if there are no predictions for label.
        LOG(FATAL)
            << "Could not find confidence predictions for " << label;
        continue;
      }
      const vector<BBoxMatch<Dtype> >& scores =
          conf_scores.find(label)->second;
      int loc_label = share_location_ ? -1 : label;
      if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for label.
        LOG(FATAL)
            << "Could not find location predictions for " << loc_label;
        continue;
      }
      const vector<NormalizedBBox>& bboxes =
          decode_bboxes.find(loc_label)->second;
      const vector<int>& indices = it->second;

      for (int j = 0; j < indices.size(); ++j) {
        const int idx = indices[j];
        const NormalizedBBox& bbox = bboxes[idx];

        top_data[count * record_size] = i;
        top_data[count * record_size + 1] = label;
        top_data[count * record_size + 2] = scores[idx].conf;
        top_data[count * record_size + 3] = bbox.xmin();
        top_data[count * record_size + 4] = bbox.ymin();
        top_data[count * record_size + 5] = bbox.xmax();
        top_data[count * record_size + 6] = bbox.ymax();
        top_data[count * record_size + 7] = scores[idx].anchor;
        top_data[count * record_size + 8] = scores[idx].x_cell_pos;
        top_data[count * record_size + 9] = scores[idx].y_cell_pos;

        ++count;
      }
    }
  }
}

INSTANTIATE_CLASS(DetectionOutputLayer);
REGISTER_LAYER_CLASS(DetectionOutput);

}  // namespace caffe
