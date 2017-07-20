#ifndef _C_CAFFE_CUSTOM_LAYER_H_
#define _C_CAFFE_CUSTOM_LAYER_H_

#include "Common.h"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"

#include "CommonDefinitions.h"

#include "Exports.h"
#include <map>
#include <vector>

using namespace caffe;
using namespace std;

#define DECLARE_CUSTOM_LAYER(classname, baseclassname, layername) \
class classname : public CustomLayer, public baseclassname<float> \
{ \
public: \
	explicit classname(const LayerParameter& param); \
	virtual inline const char* type() const  { return layername;  }  \
virtual const char* getName(); \
protected: \
	void Forward_cpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top); \
}; \


#define INSTANTIATE_CUSTOM_LAYER(classname, baseclassname,layername) \
classname::classname(const LayerParameter& param) : CustomLayer(), baseclassname<float>(param) \
{ \
	\
} \
	\
void classname::Forward_cpu(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top) \
{ \
	bool runNative = BaseCallback(m_forwardCallback, bottom, top); \
	if (runNative) \
		baseclassname<float>::Forward_cpu(bottom, top); \
} \
\
const char* classname::getName() \
{ \
	return layer_param().name().c_str(); \
} \
	\
boost::shared_ptr<Layer<float> > Get##classname(const LayerParameter& param) \
{ \
	return boost::shared_ptr<Layer<float> >(new classname(param)); \
} \
	\
static LayerRegisterer<float> g_creator_f_##classname(layername, Get##classname);     \


class CustomLayer
{
public:
	CustomLayer();
	void SetCallbacks(Callback forwardCallback);
	virtual ~CustomLayer() = 0;
protected:
	bool BaseCallback(Callback callback, const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
	virtual const char* getName() = 0;
	Callback m_forwardCallback;
};


DECLARE_CUSTOM_LAYER(CustomConvolutionLayer, ConvolutionLayer, CUSTOM_CONVOLUTION_LAYER_NAME)
DECLARE_CUSTOM_LAYER(CustomPoolingLayer, PoolingLayer, CUSTOM_POOLING_LAYER_NAME)
DECLARE_CUSTOM_LAYER(CustomDeconvolutionLayer, DeconvolutionLayer, CUSTOM_DECONVOLUTION_LAYER_NAME)
DECLARE_CUSTOM_LAYER(CustomInnerProductLayer, InnerProductLayer, CUSTOM_INNER_PRODUCT_LAYER_NAME)
DECLARE_CUSTOM_LAYER(CustomLRNLayer, LRNLayer, CUSTOM_LRN_LAYER_NAME)
DECLARE_CUSTOM_LAYER(CustomRELULayer, ReLULayer, CUSTOM_RELU_LAYER_NAME)
DECLARE_CUSTOM_LAYER(CustomSoftmaxLayer, SoftmaxLayer, CUSTOM_SOFTMAX_LAYER_NAME)
DECLARE_CUSTOM_LAYER(CustomTanHLayer, TanHLayer, CUSTOM_TANH_LAYER_NAME)

#endif