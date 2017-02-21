#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/psroi_assembling_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void PSROIAssemblingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int input_channels_num,
    const int height, const int width,
    const Dtype* bottom_rois,
    const int output_channels_num,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel,
    Dtype* top_mask_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int w = index % width;
      int h = (index / width) % height;
      int ctop = (index / width / height) % output_channels_num;
      int n = index / width / height / output_channels_num;

      // [start, end) interval for spatial sampling
      const Dtype* bottom_roi = bottom_rois + n * 5;
      int roi_batch_ind = bottom_roi[0];
      Dtype roi_start_w =
          static_cast<Dtype>(round(bottom_roi[1])) * spatial_scale;
      Dtype roi_start_h =
          static_cast<Dtype>(round(bottom_roi[2])) * spatial_scale;
      Dtype roi_end_w =
          static_cast<Dtype>(round(bottom_roi[3]) + 1.) * spatial_scale;
      Dtype roi_end_h =
          static_cast<Dtype>(round(bottom_roi[4]) + 1.) * spatial_scale;

      // If current pixel doesn't belong to ROI, ignore it.
      if (h < roi_start_h || h >= roi_end_h || w < roi_start_w ||
          w >= roi_end_w) {
        continue;
      }

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, Dtype(0.1));  // avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, Dtype(0.1));

      // Compute w and h at bottom
      Dtype bin_height = roi_height / static_cast<Dtype>(group_size);
      Dtype bin_width = roi_width / static_cast<Dtype>(group_size);
      int bin_h = static_cast<int>(floor((h - roi_start_h) / bin_height));
      int bin_w = static_cast<int>(floor((w - roi_start_w) / bin_width));

      int c = (ctop * group_size + bin_h) * group_size + bin_w;

      int bottom_index = ((roi_batch_ind * input_channels_num + c) * height +
                          h) * width + w;
      top_data[index] = bottom_data[bottom_index];
      mapping_channel[index] = c;
      if (top_mask_data != NULL) {
        int mask_index = (n * height + h) * width + w;
        top_mask_data[mask_index] = 2;
      }
    }
  }

  template <typename Dtype>
  void PSROIAssemblingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    Dtype* top_mask_data = NULL;
    if (top.size() == 2) {
      top_mask_data = top[1]->mutable_gpu_data();
      caffe_gpu_set(count, Dtype(0), top_mask_data);
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIAssemblingForward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_,
      input_channels_num_, height_, width_,
      bottom_rois, output_channels_num_, group_size_,
      top_data, mapping_channel_ptr, top_mask_data);
    CUDA_POST_KERNEL_CHECK;
  }

  template <typename Dtype>
  __global__ void PSROIAssemblingBackward(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int input_channels_num,
    const int height, const int width,
    const int output_channels_num,
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int c = mapping_channel[index];
      if (c == -1) {
        continue;
      }

      int w = index % width;
      int h = (index / width) % height;
      int n = index / width / height / output_channels_num;
      const Dtype* bottom_roi = bottom_rois + n * 5;
      // bottom_rois += n * 5;
      int roi_batch_ind = bottom_roi[0];
      int bottom_index = ((roi_batch_ind * input_channels_num + c) * height +
                          h) * width + w;
      bottom_diff[bottom_index] += top_diff[index];
    }
  }

  template <typename Dtype>
  void PSROIAssemblingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    PSROIAssemblingBackward<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr,
      input_channels_num_, height_, width_,
      output_channels_num_, bottom_diff, bottom_rois);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(PSROIAssemblingLayer);

}  // namespace caffe
