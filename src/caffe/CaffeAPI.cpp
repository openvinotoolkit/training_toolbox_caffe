#include "caffe/CaffeAPI.h"
#include "caffe/CaffeCustomLayer.h"

#include <stdio.h>
#include <stdlib.h>

#include "caffe/net.hpp"
#include <cstdlib>

using namespace caffe;


#define REGISTER_FLOAT_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator);     \


#ifdef __cplusplus
extern "C" {
#endif

net_wrapper createNet_(proto_descriptor topology)
{
	_net_wrapper* desc = new _net_wrapper();
	Net<float>* net = new Net<float>(*((NetParameter*)topology->param));
	desc->param = net;
	return (net_wrapper)desc;
}

void releaseNet_(net_wrapper network)
{
	delete TO_CAFFE_NET(network->param);
	delete network;
}

void initLogging_()
{
	int count = 1;
	char* cmd = (char*)"--logtostderr=1";
	char** argv = { &cmd };
	GlobalInit(&count, &argv);
}

const char* getNetName_(net_wrapper network)
{
	return TO_CAFFE_NET(network->param)->name().c_str();
}

const int* getTopBlobsIds_(net_wrapper network,int layerIndex, int& count)
{
	const vector<int>& topIds = TO_CAFFE_NET(network->param)->top_ids(layerIndex);
	count = topIds.size();
	return topIds.data();
}

const int* getBottomBlobsIds_(net_wrapper network, int layerIndex, int& count)
{
	const vector<int>& topIds = TO_CAFFE_NET(network->param)->bottom_ids(layerIndex);
	count = topIds.size();
	return topIds.data();
}

const int* getOutputBlobsIds_(net_wrapper network, int& count)
{
	const vector<int>& bottomIds = TO_CAFFE_NET(network->param)->output_blob_indices();
	count = bottomIds.size();
	return bottomIds.data();
}

const char* getBlobNames_(net_wrapper network, int index)
{
	const vector<string>& blobsNames = TO_CAFFE_NET(network->param)->blob_names(); 
	return blobsNames[index].c_str();
}

int getNetworkLayersCount_(net_wrapper network)
{
	return TO_CAFFE_NET(network->param)->layers().size();
}

void loadWeights_(net_wrapper network, proto_descriptor weights)
{
	TO_CAFFE_NET(network->param)->CopyTrainedLayersFrom(*((NetParameter*)weights->param));
}

void** forwardWithLoss_(net_wrapper network, float* loss, int& count)
{
	const vector<Blob<float>*>& blobs = TO_CAFFE_NET(network->param)->Forward(loss);
	count = blobs.size();
	void** ptr = (void**)(blobs.data());
	return ptr;
}

void forward_(net_wrapper network)
{
	TO_CAFFE_NET(network->param)->Forward();
}

void reshape_(net_wrapper network)
{
	TO_CAFFE_NET(network->param)->Reshape();
}

layer_wrapper getLayer_(net_wrapper network, int index)
{
	const vector<boost::shared_ptr<Layer<float> > >& layers = TO_CAFFE_NET(network->param)->layers();
	Layer<float>* layer = layers[index].get();
	_layer_wrapper* desc = new _layer_wrapper();
	desc->param = (void*)layer;
	return (layer_wrapper)desc;
}

layer_wrapper createLayer_(void* layer)
{
	Layer<float>* nativeLayer = TO_CAFFE_LAYER(layer);
	_layer_wrapper* desc = new _layer_wrapper();
	desc->param = nativeLayer;
	return (layer_wrapper)desc;
}

blob_wrapper createBlob_(void* blob)
{
	Blob<float>* nativeParam= TO_CAFFE_BLOB(blob);
	blob_wrapper desc = new _blob_wrapper();
	desc->param = nativeParam;
	return (blob_wrapper)desc;
}

void releaseBlob_(blob_wrapper blob)
{
	delete blob;
}

void releaseLayer_(layer_wrapper layer)
{
	delete layer;
}

const char* getLayerType_(layer_wrapper layer)
{
	return TO_CAFFE_LAYER(layer->param)->type();
}

const char* getLayerName_(layer_wrapper layer)
{
	return TO_CAFFE_LAYER(layer->param)->layer_param().name().c_str();
}

float forwardLayer_(net_wrapper net, layer_wrapper layer, int layerIndex)
{
	Net<float>* nativeNet = TO_CAFFE_NET(net->param);
	Layer<float>* nativeLayer = TO_CAFFE_LAYER(layer->param);
	const vector<Blob<float>*>& bottom = nativeNet->bottom_vecs()[layerIndex];
	const vector<Blob<float>*>& top = nativeNet->top_vecs()[layerIndex];
	return nativeLayer->Forward(bottom, top);
}


int getNetworkBlobsCount_(net_wrapper layer)
{
	return TO_CAFFE_NET(layer->param)->blobs().size();
}

int getLayerBlobsCount_(layer_wrapper layer)
{
	return TO_CAFFE_LAYER(layer->param)->blobs().size();
}

blob_wrapper getNetworkBlob_(net_wrapper network, int index)
{
	Blob<float>* blob = (TO_CAFFE_NET(network->param)->blobs()[index].get());
	_blob_wrapper* desc = new _blob_wrapper();
	desc->param = blob;
	return (blob_wrapper)desc;
}

blob_wrapper getLayerBlob_(layer_wrapper network, int index)
{
	Blob<float>* blob = (TO_CAFFE_LAYER(network->param)->blobs()[index].get());
	_blob_wrapper* desc = new _blob_wrapper();
	desc->param = blob;
	return (blob_wrapper)desc;
}

bool isCustomLayer_(layer_wrapper layer)
{
	Layer<float>* cafe_layyer = TO_CAFFE_LAYER(layer->param);
	CustomLayer* custom = dynamic_cast<CustomLayer*>(cafe_layyer);
	return (custom != NULL);
}

void setCustomLayerCallbacks_(layer_wrapper layer, Callback forward)
{
	Layer<float>* cafe_layyer = TO_CAFFE_LAYER(layer->param);
	CustomLayer* custom = dynamic_cast<CustomLayer*>(cafe_layyer);
	custom->SetCallbacks(forward);
}




#ifdef __cplusplus
}
#endif