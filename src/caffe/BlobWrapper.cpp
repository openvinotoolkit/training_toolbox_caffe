#include "caffe/CaffeAPI.h"

#include <stdio.h>
#include <stdlib.h>
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include <cstdlib>

using namespace caffe;

#ifdef __cplusplus
extern "C" {
#endif

int getCount_(blob_wrapper desc)
{
	return TO_CAFFE_BLOB(desc->param)->count();
}

float* getData_(blob_wrapper desc)
{
	return TO_CAFFE_BLOB(desc->param)->mutable_cpu_data();
}

const int* getShape_(blob_wrapper desc, int& shapeCount)
{
	const vector<int>& shape = TO_CAFFE_BLOB(desc->param)->shape();
	shapeCount = shape.size();
	return shape.data();
}

int getNumAxis_(blob_wrapper desc)
{
	return TO_CAFFE_BLOB(desc->param)->num_axes();
}

blob_wrapper createFromBinaryFile(const char* path)
{
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(path, &blob_proto);
	Blob<float>* blob = new Blob<float>();
	blob->FromProto(blob_proto);

	blob_wrapper desc = new _blob_wrapper();
	desc->param = blob;
	return (blob_wrapper)desc;

}

#ifdef __cplusplus
}
#endif