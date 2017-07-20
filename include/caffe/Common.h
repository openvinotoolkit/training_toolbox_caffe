#ifndef _C_COMMON_
#define _C_COMMON_

#define TO_NET_PARAM(param) \
	((NetParameter*)(param))

#define TO_LAYER_PARAM(param) \
	((LayerParameter*)(param))

#define TO_CAFFE_NET(param) \
	((Net<float>*)(param))

#define TO_CAFFE_BLOB(param) \
	((Blob<float>*)(param))

#define TO_CAFFE_LAYER(param) \
	((Layer<float>*)(param))


#endif