#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/reorg_yolo_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ReorgYoloLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ReorgYoloLayerTest() {
  }

  virtual void SetUp() {
  }

  virtual ~ReorgYoloLayerTest() {
  }

};

TYPED_TEST_CASE(ReorgYoloLayerTest, ::testing::Types<CPUDevice<float> >);

TYPED_TEST(ReorgYoloLayerTest, onDataFromDarknet) {
  LayerParameter layer_param;
  ReorgYoloParameter *reorg_yolo_param = layer_param.mutable_reorg_yolo_param();
  const int stride = 2;

  const size_t batch = 1;
  const size_t width = 26, height = 26, channels = 64;
  const size_t count = batch * width * height * channels;

  reorg_yolo_param->set_stride(stride);
  std::ifstream in(CMAKE_SOURCE_DIR "caffe/test/test_data/yolov2/reorg_in.data", std::ios::binary);
  ASSERT_TRUE(in.is_open());
  float *inBuf = new float[count];
  in.read((char *)inBuf, count * sizeof(float));
  in.close();

  in.open(CMAKE_SOURCE_DIR "caffe/test/test_data/yolov2/reorg_ref.data", std::ios::binary);
  ASSERT_TRUE(in.is_open());
  float *outBufRef = new float[count];
  in.read((char *)outBufRef, count * sizeof(float));
  in.close();

  vector<Blob<float> *> bottom;
  Blob<float> *bottomBlob = new Blob<float>(batch, channels,width, height);
  bottomBlob->set_cpu_data(inBuf);
  bottom.push_back(bottomBlob);

  vector<Blob<float> *> top;
  vector<int> topShape;
  topShape.push_back(1);
  topShape.push_back(channels * stride * stride);
  topShape.push_back(height / stride);
  topShape.push_back(width  / stride);
  Blob<float> *topBlob = new Blob<float>(topShape);
  top.push_back(topBlob);

  ReorgYoloLayer<float> layer(layer_param);
  layer.SetUp(bottom, top);
  layer.Forward(bottom, top);

  for (int i = 0; i < count; i++) {
    ASSERT_FLOAT_EQ(outBufRef[i], topBlob->cpu_data()[i]);
  }

  delete []inBuf;
  delete []outBufRef;
}

}  // namespace caffe
