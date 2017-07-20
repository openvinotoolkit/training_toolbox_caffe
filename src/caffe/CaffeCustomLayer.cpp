#include "caffe/CaffeCustomLayer.h"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include <float.h>

using namespace caffe;
using namespace std;

CustomLayer::CustomLayer()
{
	m_forwardCallback = NULL;
}

CustomLayer::~CustomLayer()
{

}

bool CustomLayer::BaseCallback(Callback callback, const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top)
{
	int bottomSize = bottom.size();
	void** bottomPtr = (void**)(bottom.data());

	int topSize = top.size();
	void** topPtr = (void**)(top.data());
	if (callback)
		return callback(getName(), bottomPtr, bottomSize, topPtr, topSize);
	return true;
}

void CustomLayer::SetCallbacks(Callback forwardCallback)
{
	m_forwardCallback = forwardCallback;
}

INSTANTIATE_CUSTOM_LAYER(CustomConvolutionLayer, ConvolutionLayer, CUSTOM_CONVOLUTION_LAYER_NAME)
INSTANTIATE_CUSTOM_LAYER(CustomPoolingLayer, PoolingLayer, CUSTOM_POOLING_LAYER_NAME)
INSTANTIATE_CUSTOM_LAYER(CustomDeconvolutionLayer, DeconvolutionLayer, CUSTOM_DECONVOLUTION_LAYER_NAME)
INSTANTIATE_CUSTOM_LAYER(CustomInnerProductLayer, InnerProductLayer, CUSTOM_INNER_PRODUCT_LAYER_NAME)
INSTANTIATE_CUSTOM_LAYER(CustomLRNLayer, LRNLayer, CUSTOM_LRN_LAYER_NAME)
INSTANTIATE_CUSTOM_LAYER(CustomRELULayer, ReLULayer, CUSTOM_RELU_LAYER_NAME)
INSTANTIATE_CUSTOM_LAYER(CustomSoftmaxLayer, SoftmaxLayer, CUSTOM_SOFTMAX_LAYER_NAME)
INSTANTIATE_CUSTOM_LAYER(CustomTanHLayer, TanHLayer, CUSTOM_TANH_LAYER_NAME)