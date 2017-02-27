#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/psroi_assembling_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class PSROIAssemblingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PSROIAssemblingLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 18, 10, 10)),
        blob_bottom_rois_(new Blob<Dtype>(2, 5, 1, 1)),
        blob_top_data_(new Blob<Dtype>()) {
    // Fill features blob.
    int channel_step = blob_bottom_data_->offset(0, 1, 0, 0);
    for (int i = 0; i < blob_bottom_data_->channels(); ++i) {
      for (int j = 0; j < channel_step; ++j) {
        blob_bottom_data_->mutable_cpu_data()[i * channel_step + j] = i;
      }
    }

    // Fill ROIs blob.
    int i = 0;
    blob_bottom_rois_->mutable_cpu_data()[0 + 5 * i] = 0;
    blob_bottom_rois_->mutable_cpu_data()[1 + 5 * i] = 1;
    blob_bottom_rois_->mutable_cpu_data()[2 + 5 * i] = 2;
    blob_bottom_rois_->mutable_cpu_data()[3 + 5 * i] = 3;
    blob_bottom_rois_->mutable_cpu_data()[4 + 5 * i] = 4;
    i = 1;
    blob_bottom_rois_->mutable_cpu_data()[0 + 5 * i] = 0;
    blob_bottom_rois_->mutable_cpu_data()[1 + 5 * i] = 0;
    blob_bottom_rois_->mutable_cpu_data()[2 + 5 * i] = 3;
    blob_bottom_rois_->mutable_cpu_data()[3 + 5 * i] = 5;
    blob_bottom_rois_->mutable_cpu_data()[4 + 5 * i] = 8;

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_rois_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~PSROIAssemblingLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_rois_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_rois_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PSROIAssemblingLayerTest, TestDtypesAndDevices);

TYPED_TEST(PSROIAssemblingLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* roi_pooling_param =
      layer_param.mutable_psroi_pooling_param();

  roi_pooling_param->set_spatial_scale(1);
  roi_pooling_param->set_group_size(3);
  roi_pooling_param->set_output_dim(0);
  PSROIAssemblingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), this->blob_bottom_rois_->num());
  EXPECT_EQ(this->blob_top_data_->width(), this->blob_bottom_data_->width());
  EXPECT_EQ(this->blob_top_data_->height(), this->blob_bottom_data_->height());
  EXPECT_EQ(this->blob_bottom_data_->channels() /
                roi_pooling_param->group_size() /
                roi_pooling_param->group_size(),
            this->blob_top_data_->channels());
  EXPECT_EQ(
      this->blob_bottom_data_->channels() %
          (roi_pooling_param->group_size() * roi_pooling_param->group_size()),
      0);

  for (int ch = 0; ch < this->blob_top_data_->channels(); ++ch) {
    for (int i = 0; i < this->blob_top_data_->height(); ++i) {
      for (int j = 0; j < this->blob_top_data_->width(); ++j) {
        int idx = this->blob_top_data_->offset(0, ch, i, j);
        Dtype value = this->blob_top_data_->cpu_data()[idx];
        if (i < 2 || i > 4 || j < 1 || j > 3) {
          EXPECT_EQ(0, value);
        }
      }
    }
  }

  int counter = 0;
  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 2; i < 5; ++i) {
      for (int j = 1; j < 4; ++j) {
        int idx = this->blob_top_data_->offset(0, ch, i, j);
        Dtype value = this->blob_top_data_->cpu_data()[idx];
        EXPECT_EQ(counter++, value);
      }
    }
  }

  for (int ch = 0; ch < this->blob_top_data_->channels(); ++ch) {
    for (int i = 0; i < this->blob_top_data_->height(); ++i) {
      for (int j = 0; j < this->blob_top_data_->width(); ++j) {
        int idx = this->blob_top_data_->offset(1, ch, i, j);
        Dtype value = this->blob_top_data_->cpu_data()[idx];
        if (i < 3 || i > 8 || j < 0 || j > 5) {
          EXPECT_EQ(0, value);
        }
      }
    }
  }

  for (int ch = 0; ch < 2; ++ch) {
    for (int i = 3; i < 9; ++i) {
      for (int j = 0; j < 6; ++j) {
        int idx = this->blob_top_data_->offset(1, ch, i, j);
        Dtype value = this->blob_top_data_->cpu_data()[idx];
        int bin_idx = ((j - 0) / 2) + ((i - 3) / 2) * 3 + ch * 9;
        EXPECT_EQ(bin_idx, value);
      }
    }
  }
}

TYPED_TEST(PSROIAssemblingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PSROIPoolingParameter* roi_pooling_param =
      layer_param.mutable_psroi_pooling_param();
  roi_pooling_param->set_spatial_scale(1);
  roi_pooling_param->set_group_size(3);
  roi_pooling_param->set_output_dim(0);
  PSROIAssemblingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

}  // namespace caffe
