#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/region_yolo_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RegionYoloLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RegionYoloLayerTest() {

  }

  virtual void SetUp() {
  }

  virtual ~RegionYoloLayerTest() {
  }
};

TYPED_TEST_CASE(RegionYoloLayerTest, ::testing::Types<CPUDevice<float> >);

TYPED_TEST(RegionYoloLayerTest, onDataFromDarknet) {
  LayerParameter layer_param;
  RegionYoloParameter *region_yolo_param = layer_param.mutable_region_yolo_param();
  const int num = 5;
  const int coords = 4;
  const int classes = 20;
  const size_t batch = 1;
  const size_t width = 13, height = 13, channels = 125;
  const size_t count = width * height * channels;

  region_yolo_param->set_num(num);
  region_yolo_param->set_coords(coords);
  region_yolo_param->set_classes(classes);
  std::ifstream in(CMAKE_SOURCE_DIR "caffe/test/test_data/yolov2/region_in.data", std::ios::binary);
  ASSERT_TRUE(in.is_open());
  float *in_buf = new float[count];
  in.read((char *)in_buf, count * sizeof(float));
  in.close();

  in.open(CMAKE_SOURCE_DIR "caffe/test/test_data/yolov2/region_ref.data", std::ios::binary);
  ASSERT_TRUE(in.is_open());
  float *out_buf_ref = new float[count];
  in.read((char *)out_buf_ref, count * sizeof(float));
  in.close();

  vector<Blob<float> *> bottom;
  Blob<float> *bottom_blob = new Blob<float>(batch, channels,width, height);
  bottom_blob->set_cpu_data(in_buf);
  bottom.push_back(bottom_blob);

  vector<Blob<float> *> top;
  vector<int> shape;
  shape.push_back(1);
  shape.push_back(channels * width * height);
  Blob<float> *top_blob = new Blob<float>(shape);
  top.push_back(top_blob);

  RegionYoloLayer<float> layer(layer_param);
  layer.SetUp(bottom, top);
  layer.Forward(bottom,top);

  for (int i = 0; i < count; i++) {
    ASSERT_FLOAT_EQ(out_buf_ref[i], top_blob->cpu_data()[i]);
  }

  delete []in_buf;
  delete []out_buf_ref;
}

}  // namespace caffe
