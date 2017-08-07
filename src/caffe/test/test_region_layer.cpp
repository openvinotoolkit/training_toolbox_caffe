#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/region_layer.hpp"

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

TYPED_TEST_CASE(RegionLayerTest, ::testing::Types<CPUDevice<float> >);

TYPED_TEST(RegionLayerTest, onDataFromDarknet) {
  LayerParameter layer_param;
  RegionParameter *region_param = layer_param.mutable_region_param();
  const int num = 5;
  const int coords = 4;
  const int classes = 20;
  const size_t batch = 1;
  const size_t width = 13, height = 13, channels = 125;
  const size_t count = width * height * channels;

  region_param->set_num(num);
  region_param->set_coords(coords);
  region_param->set_classes(classes);
  std::ifstream in("../src/caffe/test/test_data/region_reference_data/dog_in.bin", std::ios::binary);
  float *inBuf = new float[count];
  in.read((char *)inBuf, count * sizeof(float));
  in.close();

  in.open("../src/caffe/test/test_data/region_reference_data/dog_ref_out.bin", std::ios::binary);
  float *outBufRef = new float[count];
  in.read((char *)outBufRef, count * sizeof(float));
  in.close();

  vector<Blob<float> *> bottom;
  Blob<float> *bottomBlob = new Blob<float>(batch, channels, width, height);
  bottomBlob->set_cpu_data(inBuf);
  bottom.push_back(bottomBlob);

  vector<Blob<float> *> top;
  Blob<float> *topBlob = new Blob<float>(batch, channels, width, height);
  top.push_back(topBlob);

  RegionLayer<float> layer(layer_param);
  layer.SetUp(bottom, top);
  layer.Forward(bottom,top);

  for (int i = 0; i < count; i++) {
    ASSERT_FLOAT_EQ(outBufRef[i], topBlob->cpu_data()[i]);
  }
}

}  // namespace caffe
