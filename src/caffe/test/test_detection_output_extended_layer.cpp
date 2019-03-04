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
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/detection_output_extended_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

static const float eps = 1e-6;

template <typename TypeParam>
class DetectionOutputExtendedLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DetectionOutputExtendedLayerTest()
      : num_(2),
        num_priors_(4),
        num_classes_(2),
        share_location_(true),
        num_loc_classes_(share_location_ ? 1 : num_classes_),
        background_label_id_(0),
        nms_threshold_(0.1),
        top_k_(2),
        blob_bottom_loc_(
            new Blob<Dtype>(num_, 1, 1, num_priors_ * num_loc_classes_ * 4)),
        blob_bottom_conf_(
            new Blob<Dtype>(num_, 1, 1, num_priors_ * num_classes_)),
        blob_bottom_prior_(new Blob<Dtype>(num_, 2, num_priors_ * 4, 1)),
        blob_top_(new Blob<Dtype>()) {
    // Fill prior data first.
    Dtype*  prior_data = blob_bottom_prior_->mutable_cpu_data();
    const float step = 0.5;
    const float box_size = 0.3;
    int idx = 0;
    for (int h = 0; h < 2; ++h) {
      float center_y = (h + 0.5) * step;
      for (int w = 0; w < 2; ++w) {
        float center_x = (w + 0.5) * step;
        prior_data[idx++] = (center_x - box_size / 2);
        prior_data[idx++] = (center_y - box_size / 2);
        prior_data[idx++] = (center_x + box_size / 2);
        prior_data[idx++] = (center_y + box_size / 2);
      }
    }
    for (int i = 0; i < idx; ++i) {
      prior_data[idx + i] = 0.1;
    }

    // Fill confidences.
    Dtype* conf_data = blob_bottom_conf_->mutable_cpu_data();
    idx = 0;
    for (int i = 0; i < this->num_; ++i) {
      for (int j = 0; j < this->num_priors_; ++j) {
        for (int c = 0; c < this->num_classes_; ++c) {
          if (i % 2 == c % 2) {
            conf_data[idx++] = j * 0.2;
          } else {
            conf_data[idx++] = 1 - j * 0.2;
          }
        }
      }
    }

    blob_bottom_vec_.push_back(blob_bottom_loc_);
    blob_bottom_vec_.push_back(blob_bottom_conf_);
    blob_bottom_vec_.push_back(blob_bottom_prior_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DetectionOutputExtendedLayerTest() {
    delete blob_bottom_loc_;
    delete blob_bottom_conf_;
    delete blob_bottom_prior_;
    delete blob_top_;
  }

  void FillLocData(const bool share_location = true) {
    // Fill location offsets.
    int num_loc_classes = share_location ? 1 : this->num_classes_;
    blob_bottom_loc_->Reshape(
        this->num_, 1, 1, this->num_priors_ * num_loc_classes * 4);
    Dtype* loc_data = blob_bottom_loc_->mutable_cpu_data();
    int idx = 0;
    for (int i = 0; i < this->num_; ++i) {
      for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 2; ++w) {
          for (int c = 0; c < num_loc_classes; ++c) {
            loc_data[idx++] = (w % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
            loc_data[idx++] = (h % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
            loc_data[idx++] = (w % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
            loc_data[idx++] = (h % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
          }
        }
      }
    }
  }

  void CheckEqual(const Blob<Dtype>& blob, const int num, const string values) {
    CHECK_LT(num, blob.height());

    // Split values to vector of items.
    vector<string> items;
    std::istringstream iss(values);
    std::copy(std::istream_iterator<string>(iss),
              std::istream_iterator<string>(), back_inserter(items));
    EXPECT_EQ(items.size(), 7);

    // Check data.
    const Dtype* blob_data = blob.cpu_data();
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(static_cast<int>(blob_data[num * blob.width() + i]),
                atoi(items[i].c_str()));
    }
    for (int i = 2; i < 7; ++i) {
      EXPECT_NEAR(blob_data[num * blob.width() + i],
                  atof(items[i].c_str()), eps);
    }
  }

  int num_;
  int num_priors_;
  int num_classes_;
  bool share_location_;
  int num_loc_classes_;
  int background_label_id_;
  float nms_threshold_;
  int top_k_;

  Blob<Dtype>* const blob_bottom_loc_;
  Blob<Dtype>* const blob_bottom_conf_;
  Blob<Dtype>* const blob_bottom_prior_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DetectionOutputExtendedLayerTest, TestDtypesAndDevices);

TYPED_TEST(DetectionOutputExtendedLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  DetectionOutputExtendedLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 10);
}

TYPED_TEST(DetectionOutputExtendedLayerTest, TestForwardShareLocation) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  detection_output_param->set_share_location(true);
  detection_output_param->set_background_label_id(0);
  detection_output_param->mutable_nms_param()->set_nms_threshold(
      this->nms_threshold_);
  DetectionOutputExtendedLayer<Dtype> layer(layer_param);

  this->FillLocData(true);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 10);

  this->CheckEqual(*(this->blob_top_), 0, "0 1 1.0 0.15 0.15 0.45 0.45");
  this->CheckEqual(*(this->blob_top_), 1, "0 1 0.8 0.55 0.15 0.85 0.45");
  this->CheckEqual(*(this->blob_top_), 2, "0 1 0.6 0.15 0.55 0.45 0.85");
  this->CheckEqual(*(this->blob_top_), 3, "0 1 0.4 0.55 0.55 0.85 0.85");
  this->CheckEqual(*(this->blob_top_), 4, "1 1 0.6 0.45 0.45 0.75 0.75");
  this->CheckEqual(*(this->blob_top_), 5, "1 1 0.0 0.25 0.25 0.55 0.55");
}

TYPED_TEST(DetectionOutputExtendedLayerTest, TestForwardShareLocationTopK) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DetectionOutputParameter* detection_output_param =
      layer_param.mutable_detection_output_param();
  detection_output_param->set_num_classes(this->num_classes_);
  detection_output_param->set_share_location(true);
  detection_output_param->set_background_label_id(0);
  detection_output_param->mutable_nms_param()->set_nms_threshold(
      this->nms_threshold_);
  detection_output_param->mutable_nms_param()->set_top_k(this->top_k_);
  DetectionOutputExtendedLayer<Dtype> layer(layer_param);

  this->FillLocData(true);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 10);

  this->CheckEqual(*(this->blob_top_), 0, "0 1 1.0 0.15 0.15 0.45 0.45");
  this->CheckEqual(*(this->blob_top_), 1, "0 1 0.8 0.55 0.15 0.85 0.45");
  this->CheckEqual(*(this->blob_top_), 2, "1 1 0.6 0.45 0.45 0.75 0.75");
}

}  // namespace caffe
