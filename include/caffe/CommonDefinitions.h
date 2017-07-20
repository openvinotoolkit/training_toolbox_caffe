#ifndef _C_COMMON_DEFINITIONS_
#define _C_COMMON_DEFINITIONS_

#define CUSTOM_CONVOLUTION_LAYER_NAME "CustomConvolution"
#define CUSTOM_POOLING_LAYER_NAME "CustomPooling"
#define CUSTOM_DECONVOLUTION_LAYER_NAME "CustomDeconvolution"
#define CUSTOM_INNER_PRODUCT_LAYER_NAME "CustomInnerProduct"
#define CUSTOM_LRN_LAYER_NAME "CustomLRN"
#define CUSTOM_RELU_LAYER_NAME "CustomRelU"
#define CUSTOM_SOFTMAX_LAYER_NAME "CustomSoftmax"
#define CUSTOM_SOFTMAX_WITH_LOSS_LAYER_NAME "CustomSoftmaxWithLoss"
#define CUSTOM_TANH_LAYER_NAME "CustomTanH"

typedef bool(*Callback)(const char* layerName, void** bottoms, int bottomCount, void** tops, int topCount);

#define MAX_SHAPE_SIZE 6
#define MAX_STRING_SIZE 256
#define MAX_PARAMS_SIZE 64
#define MAX_REPEATED_SIZE 16

enum pool_type { AVG, MAX, STOCH };
enum round_type { CIEL, FLOOR };
enum norm_type { SAME_CHANNEL, ACROSS_CHANNELS };
enum operation_type { OP_MUL, OP_MAX, OP_SUM };

enum caffe_param_type { INT32_DT, UINT32_DT, INT64_DT, UINT64_DT, FLOAT_DT, BOOL_DT, DOUBLE_DT, ENUM_DT, STRING_DT, REPEATED };


typedef struct _layer_descriptor* layer_descriptor;
typedef struct _convolution_descriptor* convolution_descriptor;
typedef struct _data_layer_descriptor* data_layer_descriptor;
typedef struct _input_layer_descriptor* input_layer_descriptor;
typedef struct _crop_layer_descriptor* crop_layer_descriptor;
typedef struct _inner_product_layer_descriptor* inner_product_layer_descriptor;
typedef struct _pooling_layer_descriptor* pooling_layer_descriptor;
typedef struct _lrn_layer_descriptor* lrn_layer_descriptor;
typedef struct _softmax_layer_descriptor* softmax_layer_descriptor;
typedef struct  _concat_layer_descriptor* concat_layer_descriptor;
typedef struct  _power_layer_descriptor* power_layer_descriptor;
typedef struct  _element_wise_layer_descriptor* element_wise_layer_descriptor;
typedef struct  _batch_norm_descriptor* batch_norm_descriptor;
typedef struct _generic_layer_descriptor*  generic_layer_descriptor;
typedef struct _meta_data_descriptor*  meta_data_descriptor;

typedef struct _proto_descriptor* proto_descriptor;
typedef struct _net_wrapper* net_wrapper;
typedef struct _layer_wrapper* layer_wrapper;
typedef struct _blob_wrapper* blob_wrapper;


typedef struct _net_wrapper
{
	void* param;

} _net_wrapper_t;


typedef struct _layer_wrapper
{
	void* param;

} _layer_wrapper_t;

typedef struct _blob_wrapper
{
	void* param;

} _blob_wrapper_t;


typedef struct _proto_descriptor
{
	void* param;

} _proto_descriptor_t;

typedef struct _layer_descriptor
{
	char* name;
	char* type;
	void* modelParam; //weights param
	void* dataParam; //topology param
	bool isTestPhase;
} layer_descriptor_t;

typedef struct _caffe_layer_metadata
{
	caffe_param_type type;
	char param_name[MAX_STRING_SIZE];

} _caffe_layer_metadata_t;

typedef struct _caffe_metadata_simple_value
{
	char param_value[MAX_STRING_SIZE];

} _caffe_metadata_simple_value_t;

typedef struct _caffe_metadata_simple_type
{
	_caffe_layer_metadata_t base;
	_caffe_metadata_simple_value_t val;
} _caffe_metadata_simple_type_t;

typedef struct _caffe_repeated_type
{
	_caffe_layer_metadata_t base;
	_caffe_metadata_simple_value_t simple_type[MAX_REPEATED_SIZE];
	int count;
	caffe_param_type cell_type;
} _caffe_repeated_type_t;

typedef struct _meta_data_descriptor
{
	layer_descriptor_t base;
	_caffe_layer_metadata_t* metaData[MAX_PARAMS_SIZE];
	int params_count;

} _meta_data_descriptor_t;


typedef struct _convolution_descriptor
{
	layer_descriptor_t base;

	unsigned int stride_x;
	unsigned int stride_y;
	unsigned int pad_x;
	unsigned int pad_y;
	unsigned int kernel_x;
	unsigned int kernel_y;
	unsigned int output_size;
	unsigned int group;
	bool bias_term;

	unsigned int dilatation_x;
	unsigned int dilatation_y;
	bool hasDilation;

} convolution_descriptor_t;


typedef struct _data_layer_descriptor
{
	layer_descriptor_t base;

	int batchSize;

} data_layer_descriptor_t;

typedef struct _input_layer_descriptor
{
	layer_descriptor_t base;
	int shape[MAX_SHAPE_SIZE];
	int shapeSize;

} input_layer_descriptor_t;


typedef struct _crop_layer_descriptor
{
	layer_descriptor_t base;
	int offsets[MAX_SHAPE_SIZE];
	int offsetsSize;
	int axis;

} crop_layer_descriptor_t;


typedef struct _inner_product_layer_descriptor
{
	layer_descriptor_t base;
	int num_outputs;
	bool bias_term;

} inner_product_layer_descriptor_t;

typedef struct _pooling_layer_descriptor
{
	layer_descriptor_t base;

	unsigned int stride_x;
	unsigned int stride_y;
	unsigned int pad_x;
	unsigned int pad_y;
	unsigned int kernel_x;
	unsigned int kernel_y;
	pool_type methodType;
	round_type roundType;
	bool isGlobalPooling;

} pooling_layer_descriptor_t;

typedef struct _lrn_layer_descriptor
{
	layer_descriptor_t base;

	float alpha;
	float beta;
	int local_size;
	norm_type normType;

} lrn_layer_descriptor_t;

typedef struct _softmax_layer_descriptor
{
	layer_descriptor_t base;
	int axis;

} softmax_layer_descriptor_t;

typedef struct _concat_layer_descriptor
{
	layer_descriptor_t base;
	int axis;

} concat_layer_descriptor_t;

typedef struct _power_layer_descriptor
{
	layer_descriptor_t base;
	float power;
	float scale;
	float shift;

} power_layer_descriptor_t;

typedef struct _element_wise_layer_descriptor
{
	layer_descriptor_t base;
	operation_type op_type;

} element_wise_layer_descriptor_t;

typedef struct _batch_norm_descriptor
{
	layer_descriptor_t base;
	float eps;

} batch_norm_descriptor_t;


typedef struct _generic_layer_descriptor
{
	layer_descriptor_t base;

} generic_layer_descriptor_t;

#endif