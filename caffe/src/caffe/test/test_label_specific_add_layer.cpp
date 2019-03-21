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
#include "caffe/layers/label_specific_add_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class LabelSpecificAddLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  LabelSpecificAddLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(5, 3, 1, 1)),
        blob_bottom_labels_(new Blob<Dtype>(5, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    Dtype *labels_data = blob_bottom_labels_->mutable_cpu_data();
    for (int i = 0; i < 5; ++i) {
      labels_data[i] = i % 3;
    }

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_labels_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~LabelSpecificAddLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_labels_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_labels_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LabelSpecificAddLayerTest, TestDtypesAndDevices);

TYPED_TEST(LabelSpecificAddLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_label_specific_add_param()->set_bias(Dtype(1.0));

  LabelSpecificAddLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(LabelSpecificAddLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype bias = Dtype(1);
  LayerParameter layer_param;
  layer_param.mutable_label_specific_add_param()->set_bias(bias);

  LabelSpecificAddLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_vec_[0]->shape(0), 5);
  EXPECT_EQ(this->blob_top_vec_[0]->shape(1), 3);
  EXPECT_EQ(this->blob_top_vec_[0]->count(), 5 * 3);

  const Dtype *input_data = this->blob_bottom_data_->cpu_data();
  const Dtype *input_labels = this->blob_bottom_labels_->cpu_data();
  const Dtype *output_data = this->blob_top_->cpu_data();
  for (int i = 0; i < 5; ++i) {
    const int label = input_labels[i];
    for (int j = 0; j < 3; ++j) {
      Dtype trg_value = input_data[i * 3 + j];
      if (j == label && trg_value > bias) {
        trg_value -= bias;
      }

      EXPECT_NEAR(output_data[i * 3 + j], trg_value, Dtype(1e-6));
    }
  }
}

TYPED_TEST(LabelSpecificAddLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_label_specific_add_param()->set_bias(Dtype(1.0));

  LabelSpecificAddLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);

  checker.CheckGradientExhaustive(
        &layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

}  // namespace caffe

