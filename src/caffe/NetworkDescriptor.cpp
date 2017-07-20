#include "caffe/CaffeAPI.h"

#include <stdio.h>
#include <stdlib.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include <cstdlib>

using namespace caffe;
using namespace std;

bool isTestPhase_(const LayerParameter& layer)
{
	//We have trouble if we accept multiple layers with the same name with different phases
	if (layer.include_size() > 0)
	{
		caffe::Phase netPhase = layer.include(0).phase();
		if (netPhase != caffe::TEST)
			return false;
	}
	return true;
}

layer_descriptor fillLayerDescriptor_(LayerParameter& layer, int size)
{
	layer_descriptor ref = (layer_descriptor)calloc(1, size);

	ref->name = (char*)layer.name().c_str();
	ref->type = (char*)layer.type().c_str();
	ref->dataParam = &layer;
	ref->isTestPhase = isTestPhase_(layer);

	return ref;
}


#ifdef __cplusplus
extern "C" {
#endif

	void recursiveSetData(const Message & mfield, meta_data_descriptor generic_desc, int& count);
	bool setSimpleType(const Message & mfield, const google::protobuf::FieldDescriptor *field, google::protobuf::FieldDescriptor::CppType cppType, const google::protobuf::Reflection *refl, string& val);
	bool setRepeatedSimpleType(const Message & mfield, const google::protobuf::FieldDescriptor *field, google::protobuf::FieldDescriptor::CppType cppType, const google::protobuf::Reflection *refl, string& val, caffe_param_type& type, int index);
	enum caffe_param_type convertProtoSimpleType(google::protobuf::FieldDescriptor::CppType cppType);

	proto_descriptor createNetworkDescriptor_()
	{
		_proto_descriptor* desc = new _proto_descriptor();
		NetParameter* param = new NetParameter();
		desc->param = (void*)param;
		return (proto_descriptor)desc;
	}

	void readTopology_(proto_descriptor desc, const char* path)
	{
		ReadNetParamsFromTextFileOrDie(path, TO_NET_PARAM(desc->param));
		TO_NET_PARAM(desc->param)->mutable_state()->set_phase(TEST);
	}

	void readWeights_(proto_descriptor desc, const char* path)
	{
		ReadNetParamsFromBinaryFileOrDie(path, TO_NET_PARAM(desc->param));
	}

	void* newLayerParam_(proto_descriptor desc)
	{
		return TO_NET_PARAM(desc->param)->add_layer();
	}

	const char* getNetworkName_(proto_descriptor desc)
	{
		return TO_NET_PARAM(desc->param)->name().c_str();
	}

	void setNetworkName_(proto_descriptor desc, const char* newName)
	{
		TO_NET_PARAM(desc->param)->set_name(newName);
	}

	void releaseNetworkDescriptor_(proto_descriptor desc)
	{
		_proto_descriptor_t* native = (_proto_descriptor_t*)(desc);
		if (native->param != NULL)
		{
			delete TO_NET_PARAM(desc->param);
			native->param = NULL;
		}

		delete native;
	}

	int getLayersCount_(proto_descriptor desc)
	{
		return TO_NET_PARAM(desc->param)->layer_size();
	}

	const char* getLayerDescriptorName_(proto_descriptor desc, int index)
	{
		return TO_NET_PARAM(desc->param)->layer(index).name().c_str();
	}

	const char* getLayerDescriptorType_(proto_descriptor desc, int index)
	{
		return TO_NET_PARAM(desc->param)->layer(index).type().c_str();
	}

	void* getLayerParam_(proto_descriptor desc, int layerIndex)
	{
		return (void*)&(TO_NET_PARAM(desc->param)->layer(layerIndex));
	}

	layer_descriptor createLayerDescriptor_(proto_descriptor desc, int layerIndex, int size)
	{
		int layersCount = getLayersCount_(desc);
		if (layerIndex > layersCount - 1)
			return NULL;

		LayerParameter& layer = const_cast<LayerParameter&>(TO_NET_PARAM(desc->param)->layer(layerIndex));

		return fillLayerDescriptor_(layer, size);
	}

	convolution_descriptor createConvolutionLayer_(proto_descriptor desc, int layerIndex)
	{
		convolution_descriptor conv_desc = (convolution_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_convolution_descriptor));

		const caffe::ConvolutionParameter& currParam = TO_LAYER_PARAM(conv_desc->base.dataParam)->convolution_param();

		int strideSize = currParam.stride_size();

		int strideX, strideY;

		if (strideSize > 0)
		{
			strideX = (strideSize > 0) ? currParam.stride(0) : 1;
			strideY = (strideSize > 1) ? currParam.stride(1) : strideX;
		}
		else
		{
			if (currParam.has_stride_h() || currParam.has_stride_w())
			{
				strideX = currParam.stride_w();
				strideY = currParam.stride_h();
			}
			else
			{
				strideX = strideY = 1; //default stride
			}
		}

		conv_desc->stride_x = strideX;
		conv_desc->stride_y = strideY;


		int padSize = currParam.pad_size();
		int padX, padY;

		if (padSize > 0)
		{
			padX = (padSize > 0) ? currParam.pad(0) : 0;
			padY = (padSize > 1) ? currParam.pad(1) : padX;
		}
		else
		{
			if (currParam.has_pad_h() || currParam.has_pad_w())
			{
				padX = currParam.pad_w();
				padY = currParam.pad_h();
			}
			else
			{
				padX = padY = 0; //default pad
			}
		}

		conv_desc->pad_x = padX;
		conv_desc->pad_y = padY;


		int kernelSize = currParam.kernel_size_size();
		int kernelX, kernelY;
		if (kernelSize > 0)
		{
			kernelX = (kernelSize > 0) ? currParam.kernel_size(0) : 0;
			kernelY = (kernelSize > 1) ? currParam.kernel_size(1) : kernelX;
		}
		else
		{
			// kernel size is mandatory field, no default values
			kernelX = currParam.kernel_w();
			kernelY = currParam.kernel_h();
		}

		conv_desc->kernel_x = kernelX;
		conv_desc->kernel_y = kernelY;

		conv_desc->output_size = currParam.num_output();
		conv_desc->group = currParam.group();
		conv_desc->bias_term = currParam.bias_term();

		int dilationSize = currParam.dilation_size();

		int dilatation_x, dilatation_y;

		if (dilationSize > 0)
		{
			dilatation_x = (dilationSize > 0) ? currParam.dilation(0) : 1;
			dilatation_y = (dilationSize > 1) ? currParam.dilation(1) : dilatation_x;
		}
		else
		{
			dilatation_x = dilatation_y = 1; //default dilation
		}

		if ((dilatation_x == 1) && (dilatation_y == 1))
		{
			conv_desc->hasDilation = false;
		}
		else
			conv_desc->hasDilation = true;

		conv_desc->dilatation_x = dilatation_x;
		conv_desc->dilatation_y = dilatation_y;

		return (convolution_descriptor)conv_desc;
	}

	data_layer_descriptor createDataLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		data_layer_descriptor data_desc = (data_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_data_layer_descriptor));

		const caffe::DataParameter& currParam = TO_LAYER_PARAM(data_desc->base.dataParam)->data_param();
		data_desc->batchSize = currParam.batch_size();
		return (data_layer_descriptor)data_desc;
	}

	input_layer_descriptor createInputLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		input_layer_descriptor data_desc = (input_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_input_layer_descriptor));

		const caffe::InputParameter& currParam = TO_LAYER_PARAM(data_desc->base.dataParam)->input_param();
		int shapeSize = currParam.shape_size();
		for (int i = 0; i < shapeSize; i++)
		{
			const caffe::BlobShape& shape = currParam.shape(i);
			for (int i = 0; i < shape.dim_size(); ++i)
			{
				data_desc->shape[i] = shape.dim(i);
			}
			data_desc->shapeSize = shape.dim_size();
			return (input_layer_descriptor)data_desc;;
		}
		return NULL;
	}

	crop_layer_descriptor createCropLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		crop_layer_descriptor crop_desc = (crop_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_crop_layer_descriptor));

		const caffe::CropParameter& currParam = TO_LAYER_PARAM(crop_desc->base.dataParam)->crop_param();

		crop_desc->axis = currParam.axis();
		int offsetSize = currParam.offset_size();

		for (int i = 0; i < offsetSize; i++)
		{
			crop_desc->offsets[i] = currParam.offset(i);
		}

		crop_desc->offsetsSize = offsetSize;

		return crop_desc;

	}

	inner_product_layer_descriptor createInnerProdcutLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		inner_product_layer_descriptor fc_desc = (inner_product_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_inner_product_layer_descriptor));
		const caffe::InnerProductParameter& currParam = TO_LAYER_PARAM(fc_desc->base.dataParam)->inner_product_param();
		fc_desc->num_outputs = currParam.num_output();
		fc_desc->bias_term = currParam.bias_term();
		return fc_desc;
	}

	pooling_layer_descriptor createPoolingLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		pooling_layer_descriptor pooling_desc = (pooling_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_pooling_layer_descriptor));
		const caffe::PoolingParameter& currParam = TO_LAYER_PARAM(pooling_desc->base.dataParam)->pooling_param();

		pooling_desc->isGlobalPooling = currParam.global_pooling();

		if (!pooling_desc->isGlobalPooling)
		{
			if (currParam.has_kernel_h())
			{
				pooling_desc->kernel_x = currParam.kernel_w();
				pooling_desc->kernel_y = currParam.kernel_h();
			}
			else
				pooling_desc->kernel_x = pooling_desc->kernel_y = currParam.kernel_size();
		}
		if (currParam.has_pad_h())
		{
			pooling_desc->pad_x = currParam.pad_w();
			pooling_desc->pad_y = currParam.pad_h();
		}
		else
			pooling_desc->pad_x = pooling_desc->pad_y = currParam.pad();

		if (currParam.has_stride_h())
		{
			pooling_desc->stride_x = currParam.stride_w();
			pooling_desc->stride_y = currParam.stride_h();
		}
		else
			pooling_desc->stride_x = pooling_desc->stride_y = currParam.stride();

		if (currParam.pool() == caffe::PoolingParameter_PoolMethod_MAX)
			pooling_desc->methodType = MAX;
		else if (currParam.pool() == caffe::PoolingParameter_PoolMethod_AVE)
			pooling_desc->methodType = AVG;
		else if (currParam.pool() == caffe::PoolingParameter_PoolMethod_STOCHASTIC)
			pooling_desc->methodType = STOCH;

		pooling_desc->roundType = CIEL; //Caffe always do ceiling!

		return pooling_desc;
	}

	lrn_layer_descriptor createLRNLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		lrn_layer_descriptor lrn_desc = (lrn_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_lrn_layer_descriptor));

		const caffe::LRNParameter& currParam = TO_LAYER_PARAM(lrn_desc->base.dataParam)->lrn_param();

		lrn_desc->alpha = currParam.alpha();
		lrn_desc->beta = currParam.beta();
		lrn_desc->local_size = currParam.local_size();

		if (currParam.norm_region() == caffe::LRNParameter_NormRegion_ACROSS_CHANNELS)
			lrn_desc->normType = ACROSS_CHANNELS;
		else if (currParam.norm_region() == caffe::LRNParameter_NormRegion_WITHIN_CHANNEL)
			lrn_desc->normType = SAME_CHANNEL;

		return lrn_desc;
	}

	softmax_layer_descriptor createSoftMaxLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		softmax_layer_descriptor sofmax_desc = (softmax_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_softmax_layer_descriptor));
		const caffe::SoftmaxParameter& currParam = TO_LAYER_PARAM(sofmax_desc->base.dataParam)->softmax_param();
		sofmax_desc->axis = currParam.axis();
		return sofmax_desc;
	}

	concat_layer_descriptor createConcatLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		concat_layer_descriptor concat_desc = (concat_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_concat_layer_descriptor));
		const caffe::ConcatParameter& currParam = TO_LAYER_PARAM(concat_desc->base.dataParam)->concat_param();
		concat_desc->axis = currParam.axis();
		return concat_desc;
	}

	power_layer_descriptor createPowerLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		power_layer_descriptor power_desc = (power_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_power_layer_descriptor));
		const caffe::PowerParameter& currParam = TO_LAYER_PARAM(power_desc->base.dataParam)->power_param();

		power_desc->power = currParam.power();
		power_desc->scale = currParam.scale();
		power_desc->shift = currParam.shift();

		return power_desc;
	}

	power_layer_descriptor createNewPowerLayerDescriptor_(const char* name, float scale, float shift, float power)
	{
		LayerParameter* newPowerLayer = new LayerParameter();
		PowerParameter* powerParam = new PowerParameter();

		powerParam->set_scale(scale);
		powerParam->set_power(power);
		powerParam->set_shift(shift);

		newPowerLayer->set_allocated_power_param(powerParam);

		string powerType = "Power";
		string powerName = name;

		newPowerLayer->set_type(powerType);
		newPowerLayer->set_name(powerName);

		newPowerLayer->add_bottom();
		newPowerLayer->add_top();

		newPowerLayer->set_top(0, powerName); //we set only the bottom name. the top will be set as the next layer top blob!

		power_layer_descriptor power_desc = (power_layer_descriptor)fillLayerDescriptor_(*newPowerLayer, sizeof(_power_layer_descriptor));

		power_desc->power = powerParam->power();
		power_desc->scale = powerParam->scale();
		power_desc->shift = powerParam->shift();

		/*LayerDescriptor* newLayer = new CaffePowerLayerDescriptor(powerName, newPowerLayer);
		newLayer->populateBlobs();*/

		return power_desc;
	}

	element_wise_layer_descriptor createElementWiseLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		element_wise_layer_descriptor element_wise_desc = (element_wise_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_element_wise_layer_descriptor));
		const caffe::EltwiseParameter& currParam = TO_LAYER_PARAM(element_wise_desc->base.dataParam)->eltwise_param();

		caffe::EltwiseParameter_EltwiseOp oper = currParam.operation();
		if (oper == caffe::EltwiseParameter_EltwiseOp_PROD)
			element_wise_desc->op_type = OP_MUL;
		else if (oper == caffe::EltwiseParameter_EltwiseOp_SUM)
			element_wise_desc->op_type = OP_SUM;
		else if (oper == caffe::EltwiseParameter_EltwiseOp_MAX)
			element_wise_desc->op_type = OP_MAX;

		return element_wise_desc;
	}

	batch_norm_descriptor createBatchNormLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		batch_norm_descriptor batch_norm_desc = (batch_norm_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_batch_norm_descriptor));
		const caffe::BatchNormParameter& currParam = TO_LAYER_PARAM(batch_norm_desc->base.dataParam)->batch_norm_param();

		batch_norm_desc->eps = currParam.eps();

		return batch_norm_desc;
	}

	generic_layer_descriptor createGenericLayerDescriptor_(proto_descriptor desc, int layerIndex)
	{
		generic_layer_descriptor generic_desc = (generic_layer_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_generic_layer_descriptor));
		return generic_desc;
	}

	meta_data_descriptor createCustomLayerDescriptor_(proto_descriptor desc, int layerIndex, const char* paramName, bool hasParam)
	{
		meta_data_descriptor generic_desc = (meta_data_descriptor)createLayerDescriptor_(desc, layerIndex, sizeof(_meta_data_descriptor));
		LayerParameter* param = TO_LAYER_PARAM(generic_desc->base.dataParam);

		if (!hasParam)
			return generic_desc;

		const google::protobuf::FieldDescriptor* currParamDesc = param->GetDescriptor()->FindFieldByName(paramName);
		const google::protobuf::Reflection *rootReflection = param->GetReflection();

		if (currParamDesc == NULL)
		{
			free(generic_desc);
			return NULL;
		}

		int count = 0;

		if (currParamDesc->type() == google::protobuf::FieldDescriptor::TYPE_MESSAGE)
		{
			const Message &mfield = rootReflection->GetMessage(*param, currParamDesc);

			recursiveSetData(mfield, generic_desc, count);
		}

		generic_desc->params_count = count;

		return generic_desc;
	}

	void recursiveSetData(const Message & mfield, meta_data_descriptor generic_desc, int& count)
	{
		const google::protobuf::Descriptor* fieldDesc = mfield.GetDescriptor();
		const google::protobuf::Reflection *refl = mfield.GetReflection();

		int fieldCount = fieldDesc->field_count();

		for (int i = 0; i < fieldCount; i++)
		{
			const google::protobuf::FieldDescriptor *field = fieldDesc->field(i);
			std::string val = "";
			caffe_param_type type;
			bool supported = true;

			google::protobuf::FieldDescriptor::CppType cppType = field->cpp_type();
			

			if (field->is_repeated())
			{
				int size = refl->FieldSize(mfield, field);
				vector<string> arr; 
				type = convertProtoSimpleType(cppType);

				for (int j = 0; j < size; j++)
				{
					supported = setRepeatedSimpleType(mfield, field, cppType, refl, val, type, j);
					if (!supported)
						break;

					arr.push_back(val);
				}
				if (supported)
				{
					_caffe_repeated_type_t* repeatedType = (_caffe_repeated_type_t*)calloc(1, sizeof(_caffe_repeated_type_t));
					repeatedType->base.type = REPEATED;
					repeatedType->cell_type = type;
					strcpy(repeatedType->base.param_name, field->name().c_str());
					repeatedType->count = arr.size();
					for (int k = 0; k < arr.size(); k++)
					{
						strcpy(repeatedType->simple_type[k].param_value, arr[k].c_str());
					}

					generic_desc->metaData[count] = (_caffe_layer_metadata_t*)repeatedType;
					count++;
				}

			}
			else
			{
				if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE)
				{
					const Message &innerMsg = refl->GetMessage(mfield, field);
					recursiveSetData(innerMsg, generic_desc, count);
					supported = false;
				}
				else
				{
					supported = setSimpleType(mfield, field, cppType, refl, val);
					type = convertProtoSimpleType(cppType);
				}

				if (supported)
				{
					_caffe_metadata_simple_type_t* simpleType = (_caffe_metadata_simple_type_t*)calloc(1, sizeof(_caffe_metadata_simple_type_t));
					simpleType->base.type = type;
					strcpy(simpleType->base.param_name, field->name().c_str());
					strcpy(simpleType->val.param_value, val.c_str());

					generic_desc->metaData[count] = (_caffe_layer_metadata_t*)simpleType;
					count++;
				}
			}
		}
	}

	enum caffe_param_type convertProtoSimpleType(google::protobuf::FieldDescriptor::CppType cppType)
	{
		caffe_param_type type;

		if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_INT32)
		{
			type = INT32_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_UINT32)
		{
			type = UINT32_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_INT64)
		{
			type = INT64_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_UINT64)
		{
			type = UINT64_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE)
		{
			type = DOUBLE_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_FLOAT)
		{
			type = FLOAT_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_BOOL)
		{
			type = BOOL_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_ENUM)
		{
			type = ENUM_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_STRING)
		{
			type = STRING_DT;
		}

		return type;
	}

	bool setSimpleType(const Message & mfield, const google::protobuf::FieldDescriptor *field, google::protobuf::FieldDescriptor::CppType cppType, const google::protobuf::Reflection *refl, string& val)
	{
		bool supported = true;
		char buffer[100];
		if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_INT32)
		{
			sprintf(buffer, "%d", refl->GetInt32(mfield, field));
			val = buffer;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_UINT32)
		{
			sprintf(buffer, "%d", refl->GetUInt32(mfield, field));
			val = buffer;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_INT64)
		{
			sprintf(buffer, "%ld", refl->GetInt64(mfield, field));
			val = buffer;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_UINT64)
		{
			sprintf(buffer, "%ld", refl->GetUInt64(mfield, field));
			val = buffer;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE)
		{
			sprintf(buffer, "%f", refl->GetDouble(mfield, field));
			val = buffer;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_FLOAT)
		{
			sprintf(buffer, "%f", refl->GetFloat(mfield, field));
			val = buffer;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_BOOL)
		{
			val = refl->GetBool(mfield, field) ? "1" : "0";
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_ENUM)
		{
			val = refl->GetEnum(mfield, field)->full_name();
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_STRING)
		{
			val = refl->GetString(mfield, field);
		}
		else
			supported = false;

		return supported;
	}


	bool setRepeatedSimpleType(const Message & mfield, const google::protobuf::FieldDescriptor *field, google::protobuf::FieldDescriptor::CppType cppType, const google::protobuf::Reflection *refl, string& val, caffe_param_type& type, int index)
	{
		bool supported = true;
		char buffer[100];

		if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_INT32)
		{
			sprintf(buffer, "%d", refl->GetRepeatedInt32(mfield, field, index));
			val = buffer;
			type = INT32_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_UINT32)
		{
			sprintf(buffer, "%d", refl->GetRepeatedUInt32(mfield, field, index));
			val = buffer;
			type = UINT32_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_INT64)
		{
			sprintf(buffer, "%ld", refl->GetRepeatedInt64(mfield, field, index));
			val = buffer;
			type = INT64_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_UINT64)
		{
			sprintf(buffer, "%ld", refl->GetRepeatedUInt64(mfield, field, index));
			val = buffer;
			type = UINT64_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE)
		{
			sprintf(buffer, "%f", refl->GetRepeatedDouble(mfield, field, index));
			val = buffer;
			type = DOUBLE_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_FLOAT)
		{
			sprintf(buffer, "%f", refl->GetRepeatedFloat(mfield, field, index));
			val = buffer;
			type = FLOAT_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_BOOL)
		{
			val = val = refl->GetRepeatedBool(mfield, field, index) ? "1" : "0";
			type = BOOL_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_ENUM)
		{
			val = refl->GetRepeatedEnum(mfield, field, index)->full_name();
			type = ENUM_DT;
		}
		else if (cppType == google::protobuf::FieldDescriptor::CPPTYPE_STRING)
		{
			val = refl->GetRepeatedString(mfield, field, index);
			type = STRING_DT;
		}
		else
			supported = false;

		return supported;
	}

	bool isTestPhase_(proto_descriptor desc, int layerIndex)
	{
		const LayerParameter& layer = TO_NET_PARAM(desc->param)->layer(layerIndex);
		//We have trouble if we accept multiple layers with the same name with different phases
		return isTestPhase_(layer);
	}

	void writeProtoToTextFile_(proto_descriptor desc, const char* name)
	{
		WriteProtoToTextFile(*(TO_NET_PARAM(desc->param)), name);
	}

	void writeProtoToBinaryFile_(proto_descriptor desc, const char* name)
	{
		WriteProtoToBinaryFile(*(TO_NET_PARAM(desc->param)), name);
	}

	bool removeLayer_(proto_descriptor desc, layer_descriptor layer)
	{
		::google::protobuf::RepeatedPtrField< ::caffe::LayerParameter >* repetedLayers = TO_NET_PARAM(desc->param)->mutable_layer();

		typedef ::google::protobuf::RepeatedPtrField<caffe::LayerParameter> RepeatedLayer;

		LayerParameter* neededParam = (LayerParameter*)layer->dataParam;

		int index = 0;
		for (RepeatedLayer::iterator it = repetedLayers->begin(); it != repetedLayers->end(); it++)
		{
			LayerParameter* currParam = &(*it);
			if (currParam == neededParam)
			{
				repetedLayers->DeleteSubrange(index, 1);
				return true;
			}
			index++;
		}
		return false;
	}

#ifdef __cplusplus
}
#endif