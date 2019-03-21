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

#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/anchor_embedding_extractor_layer.hpp"
#include "caffe/layers/grn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class AnchorEmbeddingExtractorLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  AnchorEmbeddingExtractorLayerTest()
      : blob_bottom_detections_(new Blob<Dtype>(1, 1, 5, 11)),
        blob_top_embed_(new Blob<Dtype>()),
        blob_top_labels_(new Blob<Dtype>()) {
    Dtype *blob_bottom_detections_data =
        blob_bottom_detections_->mutable_cpu_data();
    SetDetection(blob_bottom_detections_data + 0 * 11, 0, 0, 1, 0, 1, 0);
    SetDetection(blob_bottom_detections_data + 1 * 11, 0, 1, 0, 0, 0, 1);
    SetDetection(blob_bottom_detections_data + 2 * 11, 0, 2, 2, 2, 2, 1);
    SetDetection(blob_bottom_detections_data + 3 * 11, 1, 3, 3, 1, 2, 0);
    SetDetection(blob_bottom_detections_data + 4 * 11, 1, 4, 1, 2, 1, 1);
    blob_bottom_vec_.push_back(blob_bottom_detections_);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    for (int i = 0; i < 4; ++i) {
      Blob<Dtype>* new_anchor_blob = new Blob<Dtype>(2, 7, 2, 3);

      filler.Fill(new_anchor_blob);

      this->anchor_blobs_.push_back(new_anchor_blob);
      this->blob_bottom_vec_.push_back(new_anchor_blob);
    }

    blob_top_vec_.push_back(blob_top_embed_);
    blob_top_vec_.push_back(blob_top_labels_);
  }

  virtual ~AnchorEmbeddingExtractorLayerTest() {
    delete blob_bottom_detections_;
    delete anchor_blobs_[0];
    delete anchor_blobs_[1];
    delete anchor_blobs_[2];
    delete anchor_blobs_[3];
    delete blob_top_embed_;
    delete blob_top_labels_;
  }

  void SetDetection(Dtype *data, int item,  int id,
                    int anchor, int action, int x, int y) {
    data[0] = item;
    data[1] = Dtype(1);
    data[2] = Dtype(0);
    data[3] = Dtype(0);
    data[4] = Dtype(1);
    data[5] = Dtype(1);
    data[6] = anchor;
    data[7] = id;
    data[8] = action;
    data[9] = x;
    data[10] = y;
  }

  Blob<Dtype>* const blob_bottom_detections_;
  vector<Blob<Dtype>* > anchor_blobs_;
  Blob<Dtype>* const blob_top_embed_;
  Blob<Dtype>* const blob_top_labels_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(AnchorEmbeddingExtractorLayerTest, TestDtypesAndDevices);

TYPED_TEST(AnchorEmbeddingExtractorLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_embedding_extractor_param()->set_num_anchors(4);
  layer_param.mutable_embedding_extractor_param()->set_label_src_pos(8);
  layer_param.mutable_embedding_extractor_param()->set_num_valid_actions(3);

  AnchorEmbeddingExtractorLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(AnchorEmbeddingExtractorLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_embedding_extractor_param()->set_num_anchors(4);
  layer_param.mutable_embedding_extractor_param()->set_label_src_pos(8);
  layer_param.mutable_embedding_extractor_param()->set_num_valid_actions(3);

  AnchorEmbeddingExtractorLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_vec_[0]->shape(0), 5);
  EXPECT_EQ(this->blob_top_vec_[0]->shape(1), 7);
  EXPECT_EQ(this->blob_top_vec_[1]->shape(0), 5);

  const Dtype *detections_data = this->blob_bottom_detections_->cpu_data();
  const Dtype *out_data =  this->blob_top_embed_->cpu_data();
  const Dtype *out_labels =  this->blob_top_labels_->cpu_data();
  for (int i = 0; i < 5; ++i) {
    const int item = static_cast<int>(detections_data[i * 11]);
    const int anchor = static_cast<int>(detections_data[i * 11 + 6]);
    const int action = static_cast<int>(detections_data[i * 11 + 8]);
    const int x_pos = static_cast<int>(detections_data[i * 11 + 9]);
    const int y_pos = static_cast<int>(detections_data[i * 11 + 10]);

    EXPECT_EQ(action, static_cast<int>(out_labels[i]));

    const Dtype *src_data = this->anchor_blobs_[anchor]->cpu_data();
    for (int j = 0; j < 7; ++j) {
      EXPECT_NEAR(src_data[item * 7 * 2 * 3 + j * 2 * 3 + y_pos * 3 + x_pos],
                  out_data[i * 7 + j], Dtype(1e-6));
    }
  }
}

TYPED_TEST(AnchorEmbeddingExtractorLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_embedding_extractor_param()->set_num_anchors(4);
  layer_param.mutable_embedding_extractor_param()->set_label_src_pos(8);
  layer_param.mutable_embedding_extractor_param()->set_num_valid_actions(3);

  AnchorEmbeddingExtractorLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(
        &layer, this->blob_bottom_vec_, this->blob_top_vec_, 1);
  checker.CheckGradientExhaustive(
        &layer, this->blob_bottom_vec_, this->blob_top_vec_, 2);
  checker.CheckGradientExhaustive(
        &layer, this->blob_bottom_vec_, this->blob_top_vec_, 3);
  checker.CheckGradientExhaustive(
        &layer, this->blob_bottom_vec_, this->blob_top_vec_, 4);
}

}  // namespace caffe

