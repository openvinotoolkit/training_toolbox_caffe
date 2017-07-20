#ifndef _C_CAFFE_API_H_
#define _C_CAFFE_API_H_

#include "Exports.h"
#include "Common.h"
#include "CommonDefinitions.h"

#include <map>

#ifdef __cplusplus
extern "C" {
#endif
	C_API net_wrapper createNet_(proto_descriptor topology);
	C_API void releaseNet_(net_wrapper network);
	C_API void initLogging_();
	C_API const char* getNetName_(net_wrapper network);
	C_API const int* getTopBlobsIds_(net_wrapper network, int layerIndex,int& count);
	C_API const int* getBottomBlobsIds_(net_wrapper network, int layerIndex, int& count);
	C_API const int* getOutputBlobsIds_(net_wrapper network, int& count);
	C_API const char* getBlobNames_(net_wrapper network, int index);
	C_API void loadWeights_(net_wrapper network, proto_descriptor weights);

	C_API void** forwardWithLoss_(net_wrapper network, float* loss, int& count);
	C_API float forwardLayer_(net_wrapper net, layer_wrapper layer, int layerIndex);
	C_API void forward_(net_wrapper network);

	C_API void reshape_(net_wrapper network);
	C_API int getNetworkLayersCount_(net_wrapper network);
	C_API layer_wrapper getLayer_(net_wrapper network, int index);
	C_API blob_wrapper getNetworkBlob_(net_wrapper network, int index);
	C_API blob_wrapper getLayerBlob_(layer_wrapper network, int index);

	C_API layer_wrapper createLayer_(void* layer);
	C_API void releaseLayer_(layer_wrapper layer);

	C_API blob_wrapper createBlob_(void* blob);
	C_API void releaseBlob_(blob_wrapper blob);

	C_API const char* getLayerType_(layer_wrapper layer);
	C_API const char* getLayerName_(layer_wrapper layer);
	C_API int getNetworkBlobsCount_(net_wrapper layer);
	C_API int getLayerBlobsCount_(layer_wrapper layer);

	C_API bool isCustomLayer_(layer_wrapper layer);
	C_API void setCustomLayerCallbacks_(layer_wrapper layer, Callback forward);



	//Layer descriptor
	C_API void getParamsBlobData_(layer_descriptor desc, int blobIndex, float* data);
	C_API int getParamsBlobCount_(layer_descriptor desc, int blobIndex);
	C_API void getParamsBlobShape_(layer_descriptor desc, int blobIndex, int& shapeCount, int shape[MAX_SHAPE_SIZE]);

	C_API int getParamsCount_(layer_descriptor desc);
	C_API void setType_(layer_descriptor desc, const char* newType);
	C_API void setParamsBlobData_(layer_descriptor desc, int blobIndex, float* data, int count);
	C_API void clearAll_(layer_descriptor desc);
	C_API void clear_(layer_descriptor desc, int blobIndex);
	C_API void addParamBlob_(layer_descriptor desc, int size);
	C_API int getBottomSize_(layer_descriptor desc);
	C_API int getTopSize_(layer_descriptor desc);
	C_API const char* getBottomName_(layer_descriptor desc, int index);
	C_API const char* getTopName_(layer_descriptor desc, int index);
	C_API void setTopName_(layer_descriptor desc, int index, const char* name);
	C_API void setBottomName_(layer_descriptor desc, int index, const char* name);
	C_API void copyFrom_(void* from, void* to);
	C_API void releaseLayerDescriptor_(layer_descriptor desc);

	C_API void commitLRNDescriptor_(lrn_layer_descriptor desc);
	C_API void commitConvDescriptor_(convolution_descriptor desc);
	C_API void commitInnerProductDescriptor_(inner_product_layer_descriptor desc);
	C_API void commitDataDescriptor_(data_layer_descriptor desc);
	C_API void commitInputDescriptor_(input_layer_descriptor desc);

	//Network descriptor
	C_API proto_descriptor createNetworkDescriptor_();
	C_API void readTopology_(proto_descriptor desc, const char* path);
	C_API void readWeights_(proto_descriptor desc, const char* path);
	C_API void* getLayerParam_(proto_descriptor desc, int layerIndex);
	C_API void* newLayerParam_(proto_descriptor desc);
	C_API bool removeLayer_(proto_descriptor desc, layer_descriptor layer);
	C_API const char* getNetworkName_(proto_descriptor desc);
	C_API void setNetworkName_(proto_descriptor desc, const char* newName);

	C_API void writeProtoToTextFile_(proto_descriptor param, const char* name);
	C_API void writeProtoToBinaryFile_(proto_descriptor param, const char* name);

	C_API void releaseNetworkDescriptor_(proto_descriptor desc);
	C_API int getLayersCount_(proto_descriptor desc);
	C_API layer_descriptor createLayerDescriptor_(proto_descriptor desc, int layerIndex, int size);
	C_API convolution_descriptor createConvolutionLayer_(proto_descriptor desc, int layerIndex);
	C_API data_layer_descriptor createDataLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API input_layer_descriptor createInputLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API crop_layer_descriptor createCropLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API inner_product_layer_descriptor createInnerProdcutLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API pooling_layer_descriptor createPoolingLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API lrn_layer_descriptor createLRNLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API softmax_layer_descriptor createSoftMaxLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API concat_layer_descriptor createConcatLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API power_layer_descriptor createPowerLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API element_wise_layer_descriptor createElementWiseLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API batch_norm_descriptor createBatchNormLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API generic_layer_descriptor createGenericLayerDescriptor_(proto_descriptor desc, int layerIndex);
	C_API meta_data_descriptor createCustomLayerDescriptor_(proto_descriptor desc, int layerIndex, const char* paramName, bool hasParam);

	C_API power_layer_descriptor createNewPowerLayerDescriptor_(const char* name, float scale, float shift, float power);

	C_API const char* getLayerDescriptorName_(proto_descriptor desc, int index);
	C_API const char* getLayerDescriptorType_(proto_descriptor desc, int index);
	C_API bool isTestPhase_(proto_descriptor desc, int layerIndex);


	//Blob wrapper
	C_API int getCount_(blob_wrapper desc);
	C_API float* getData_(blob_wrapper desc);
	C_API const int* getShape_(blob_wrapper desc, int& shapeCount);
	C_API blob_wrapper createFromBinaryFile(const char* path);
	C_API int getNumAxis_(blob_wrapper desc);

	
#ifdef __cplusplus
}
#endif
#endif
