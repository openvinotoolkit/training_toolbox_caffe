#include <cfloat>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/psroi_assembling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
template <typename Dtype>
void PSROIAssemblingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  PSROIPoolingParameter psroi_pooling_param =
      this->layer_param_.psroi_pooling_param();
  spatial_scale_ = psroi_pooling_param.spatial_scale();
  group_size_ = psroi_pooling_param.group_size();
  CHECK_GT(group_size_, 0) << "group_size must be > 0";
}

template <typename Dtype>
void PSROIAssemblingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  input_channels_num_ = bottom[0]->channels();
  int rois_num = bottom[1]->num();
  CHECK_EQ(0, input_channels_num_ % (group_size_ * group_size_))
      << "input channel number does not match layer parameters";
  output_channels_num_ = input_channels_num_ / (group_size_ * group_size_);
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(rois_num, output_channels_num_, height_, width_);
  if (top.size() == 2) {
    top[1]->Reshape(rois_num, 1, height_, width_);
  }
  mapping_channel_.Reshape(rois_num, output_channels_num_, height_, width_);
}

template <typename Dtype>
void PSROIAssemblingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>* bottom_feature_blob = bottom[0];
  const Blob<Dtype>* bottom_rois_blob = bottom[1];
  Blob<Dtype>* top_feature_blob = top[0];

  int count = top[0]->count();

  const Dtype* bottom_data = bottom_feature_blob->cpu_data();
  const Dtype* bottom_rois = bottom_rois_blob->cpu_data();
  Dtype* top_data = top_feature_blob->mutable_cpu_data();
  Dtype* top_mask_data = NULL;
  if (top.size() == 2) {
    top_mask_data = top[1]->mutable_cpu_data();
    caffe_set(count, Dtype(0), top_mask_data);
  }
  int* mapping_channel_ptr = mapping_channel_.mutable_cpu_data();
  caffe_set(count, Dtype(0), top_data);
  caffe_set(count, -1, mapping_channel_ptr);

  for (int index = 0; index < count; ++index) {
    // The output is in order(n, ctop, ph, pw)
    int w = index % width_;
    int h = (index / width_) % height_;
    int ctop = (index / width_ / height_) % output_channels_num_;
    int n = index / width_ / height_ / output_channels_num_;

    // [start, end) interval for spatial sampling
    const Dtype* bottom_roi = bottom_rois + n * 5;
    int roi_batch_ind = bottom_roi[0];
    Dtype roi_start_w =
        static_cast<Dtype>(round(bottom_roi[1])) * spatial_scale_;
    Dtype roi_start_h =
        static_cast<Dtype>(round(bottom_roi[2])) * spatial_scale_;
    Dtype roi_end_w =
        static_cast<Dtype>(round(bottom_roi[3]) + 1.) * spatial_scale_;
    Dtype roi_end_h =
        static_cast<Dtype>(round(bottom_roi[4]) + 1.) * spatial_scale_;

    // If current pixel doesn't belong to ROI, ignore it.
    if (h < roi_start_h || h >= roi_end_h || w < roi_start_w ||
        w >= roi_end_w) {
      continue;
    }

    // Force too small ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(0.1));  // avoid 0
    Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(0.1));

    // Compute w and h at bottom
    Dtype bin_height = roi_height / static_cast<Dtype>(group_size_);
    Dtype bin_width = roi_width / static_cast<Dtype>(group_size_);
    int bin_h = static_cast<int>(floor((h - roi_start_h) / bin_height));
    int bin_w = static_cast<int>(floor((w - roi_start_w) / bin_width));

    int c = (ctop * group_size_ + bin_h) * group_size_ + bin_w;

    // int bottom_index = bottom_feature_blob->offset(roi_batch_ind, c, h, w);
    int bottom_index =
        ((roi_batch_ind * input_channels_num_ + c) * height_ + h) * width_ + w;
    CHECK_EQ(bottom_index, bottom_feature_blob->offset(roi_batch_ind, c, h, w));
    top_data[index] = bottom_data[bottom_index];
    //    LOG(INFO) << "value: " << top_data[index];
    mapping_channel_ptr[index] = c;
    if (top_mask_data != NULL) {
      int mask_index = (n * height_ + h) * width_ + w;
      top_mask_data[mask_index] = 2;
    }
  }
}

template <typename Dtype>
void PSROIAssemblingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* bottom_rois = bottom[1]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int bottom_count = bottom[0]->count();
  const int* mapping_channel = mapping_channel_.cpu_data();
  caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
  caffe_set(bottom_count, Dtype(0), bottom_diff);
  const int count = top[0]->count();

  for (int index = 0; index < count; ++index) {
    int c = mapping_channel[index];
    if (c == -1) {
      continue;
    }

    int w = index % width_;
    int h = (index / width_) % height_;
    int n = index / width_ / height_ / output_channels_num_;
    const Dtype* bottom_roi = bottom_rois + n * 5;
    // bottom_rois += n * 5;
    int roi_batch_ind = bottom_roi[0];
    int bottom_index =
        ((roi_batch_ind * input_channels_num_ + c) * height_ + h) * width_ + w;
    bottom_diff[bottom_index] += top_diff[index];
  }
}
#ifdef CPU_ONLY
STUB_GPU(PSROIAssemblingLayer);
#endif

INSTANTIATE_CLASS(PSROIAssemblingLayer);
REGISTER_LAYER_CLASS(PSROIAssembling);

}  // namespace caffe
