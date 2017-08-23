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
  float *in_buf = new float[count];
  in.read((char *)in_buf, count * sizeof(float));
  in.close();

  in.open(CMAKE_SOURCE_DIR "caffe/test/test_data/yolov2/reorg_ref.data", std::ios::binary);
  ASSERT_TRUE(in.is_open());
  float *out_buf_ref = new float[count];
  in.read((char *)out_buf_ref, count * sizeof(float));
  in.close();

  vector<Blob<float> *> bottom;
  Blob<float> *bottom_blob = new Blob<float>(batch, channels,width, height);
  bottom_blob->set_cpu_data(in_buf);
  bottom.push_back(bottom_blob);

  vector<Blob<float> *> top;
  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(channels * stride * stride);
  top_shape.push_back(height / stride);
  top_shape.push_back(width  / stride);
  Blob<float> *top_blob = new Blob<float>(top_shape);
  top.push_back(top_blob);

  ReorgYoloLayer<float> layer(layer_param);
  layer.SetUp(bottom, top);
  layer.Forward(bottom, top);

  for (int i = 0; i < count; i++) {
    ASSERT_FLOAT_EQ(out_buf_ref[i], top_blob->cpu_data()[i]);
  }

  delete []in_buf;
  delete []out_buf_ref;
}

}  // namespace caffe
