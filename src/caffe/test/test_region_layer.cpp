#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/tile_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RegionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RegionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
  }

  virtual ~RegionLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RegionLayerTest, TestDtypesAndDevices);

TYPED_TEST(RegionLayerTest, TestTrivialSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ASSERT_EQ(0, 1);
  /*const int kNumTiles = 1;
  layer_param.mutable_tile_param()->set_tiles(kNumTiles);
  for (int i = 0; i < this->blob_bottom_->num_axes(); ++i) {
    layer_param.mutable_tile_param()->set_axis(i);
    RegionLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_->num_axes(), this->blob_bottom_->num_axes());
    for (int j = 0; j < this->blob_bottom_->num_axes(); ++j) {
      EXPECT_EQ(this->blob_top_->shape(j), this->blob_bottom_->shape(j));
    }
  }*/
}
}  // namespace caffe
