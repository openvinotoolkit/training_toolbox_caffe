#include "caffe/CaffeAPI.h"

#include "caffe/proto/caffe.pb.h"

using namespace caffe;

void getParamsBlobData_(layer_descriptor desc, int blobIndex, float* data)
{
	::caffe::BlobProto blob = TO_LAYER_PARAM(desc->modelParam)->blobs(blobIndex);
	::google::protobuf::RepeatedField< float >* arr = blob.mutable_data();
	float* ptr = arr->mutable_data();
	memcpy(data, ptr, arr->size()*sizeof(float));
}

int getParamsBlobCount_(layer_descriptor desc, int blobIndex)
{
	::caffe::BlobProto blob = TO_LAYER_PARAM(desc->modelParam)->blobs(blobIndex);
	return blob.mutable_data()->size();
}

void getParamsBlobShape_(layer_descriptor desc, int blobIndex, int& shapeCount, int shapeDims[MAX_SHAPE_SIZE])
{
	::caffe::BlobProto blob = TO_LAYER_PARAM(desc->modelParam)->blobs(blobIndex);

	bool hasShape = blob.has_shape();
	
	int count = 0;

	if (hasShape)
	{
		const ::caffe::BlobShape& shape = blob.shape();
		shapeCount = shape.dim_size();
		for (int i = 0; i < shapeCount; ++i)
		{
			shapeDims[i] = shape.dim(i);
		}
	}
	else
	{
		if (blob.has_num())
		{
			shapeDims[count++] = blob.num();
		}
		if (blob.has_channels())
		{
			shapeDims[count++] = blob.channels();
		}
		if (blob.has_height())
		{
			shapeDims[count++] = blob.height();
		}
		if (blob.has_width())
		{
			shapeDims[count++] = blob.width();
		}
		shapeCount = count;
	}
	
}


int getParamsCount_(layer_descriptor desc)
{
	if (desc->modelParam == NULL)
		return -1;
	return TO_LAYER_PARAM(desc->modelParam)->blobs_size();
}

void setParamsBlobData_(layer_descriptor desc, int blobIndex, float* data, int count)
{
	::caffe::BlobProto* blob = TO_LAYER_PARAM(desc->modelParam)->mutable_blobs(blobIndex);
	blob->clear_data();
	for (int i = 0; i < count; i++)
		blob->add_data(data[i]);
}

void clearAll_(layer_descriptor desc)
{
	int size = getParamsCount_(desc);
	for (int i = 0; i < size; i++)
		clear_(desc, i);
}

void clear_(layer_descriptor desc, int blobIndex)
{
	::caffe::BlobProto* blob = TO_LAYER_PARAM(desc->modelParam)->mutable_blobs(blobIndex);
	blob->Clear();
	blob->clear_data();
}

void addParamBlob_(layer_descriptor desc, int size)
{
	::caffe::BlobProto *blob;
	if (desc->dataParam != NULL)
		blob = TO_LAYER_PARAM(desc->modelParam)->add_blobs();
	else
		return;

	::caffe::BlobShape *shape = new ::caffe::BlobShape;
	shape->add_dim(size);
	blob->set_allocated_shape(shape);
}

int getBottomSize_(layer_descriptor desc)
{
	return TO_LAYER_PARAM(desc->dataParam)->bottom_size();
}

int getTopSize_(layer_descriptor desc)
{
	return TO_LAYER_PARAM(desc->dataParam)->top_size();
}

const char* getBottomName_(layer_descriptor desc, int index)
{
	return TO_LAYER_PARAM(desc->dataParam)->bottom(index).c_str();
}

const char* getTopName_(layer_descriptor desc, int index)
{
	return TO_LAYER_PARAM(desc->dataParam)->top(index).c_str();
}

void setTopName_(layer_descriptor desc, int index, const char* name)
{
	TO_LAYER_PARAM(desc->dataParam)->set_top(index,name);
}

void setBottomName_(layer_descriptor desc, int index, const char* name)
{
	TO_LAYER_PARAM(desc->dataParam)->set_bottom(index, name);
}

void setType_(layer_descriptor desc,const char* newType)
{
	TO_LAYER_PARAM(desc->dataParam)->set_type(newType);
}

void copyFrom_(void* from, void* to)
{
	TO_LAYER_PARAM(to)->CopyFrom(*TO_LAYER_PARAM(from));
}

void releaseLayerDescriptor_(layer_descriptor desc)
{
	delete desc;
}


void commitLRNDescriptor_(lrn_layer_descriptor desc)
{
	caffe::LRNParameter& currParam = const_cast<caffe::LRNParameter&>(TO_LAYER_PARAM(desc->base.dataParam)->lrn_param());
	currParam.set_alpha(desc->alpha);
	currParam.set_beta(desc->beta);
	currParam.set_local_size(desc->local_size);
}

void commitDataDescriptor_(data_layer_descriptor desc)
{
	caffe::DataParameter& currParam = const_cast<caffe::DataParameter&>(TO_LAYER_PARAM(desc->base.dataParam)->data_param());
	currParam.set_batch_size(desc->batchSize);
}

void commitInputDescriptor_(input_layer_descriptor desc)
{
	caffe::InputParameter& currParam = const_cast<caffe::InputParameter&>(TO_LAYER_PARAM(desc->base.dataParam)->input_param());

	int shapeSize = currParam.shape_size();
	for (int i = 0; i < shapeSize; i++)
	{
		caffe::BlobShape& shape = const_cast<caffe::BlobShape&>(currParam.shape(i));
		for (int i = 0; i < shape.dim_size(); ++i)
		{
			shape.set_dim(i, desc->shape[i]); 
		}
	}
}

void commitConvDescriptor_(convolution_descriptor desc)
{
	caffe::ConvolutionParameter& currParam = const_cast<caffe::ConvolutionParameter&>(TO_LAYER_PARAM(desc->base.dataParam)->convolution_param());
	caffe::ConvolutionParameter& currModelParam = const_cast<caffe::ConvolutionParameter&>(TO_LAYER_PARAM(desc->base.modelParam)->convolution_param());

	currParam.set_bias_term(desc->bias_term);
	currModelParam.set_bias_term(desc->bias_term);

}

void commitInnerProductDescriptor_(inner_product_layer_descriptor desc)
{
	caffe::InnerProductParameter& currParam = const_cast<caffe::InnerProductParameter&>(TO_LAYER_PARAM(desc->base.dataParam)->inner_product_param());
	caffe::InnerProductParameter& currModelParam = const_cast<caffe::InnerProductParameter&>(TO_LAYER_PARAM(desc->base.modelParam)->inner_product_param());

	currParam.set_bias_term(desc->bias_term);
	currModelParam.set_bias_term(desc->bias_term);
}
