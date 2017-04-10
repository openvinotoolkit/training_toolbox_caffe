#include <vector>

#include "caffe/layers/channel_perm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ChannelPermutationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Expect single bottom.";
  CHECK_EQ(top.size(), 1) << "Expect single top.";
  CHECK_GE(bottom[0]->num_axes(), 2) << "Bottom blob must be at least 2-dimensional.";

  const ChannelPermutationParameter& param = this->layer_param_.channel_permutation_param();
  CHECK_LE(param.version(), 1) << "Need newer version of ChannelPermutation layer for this network.";

  for (int i = 0; i < param.action_size(); ++i) {
    CHECK_GE(param.action(i).chan(), 0) << "action.chan must be non-negative";
    CHECK_LT(param.action(i).chan(), param.num_output()) << "action.chan must be less than num_output";
    CHECK_EQ(int(param.action(i).has_copy()) + int(param.action(i).has_fill()), 1) << "ChannelPermutationAction must have either copy or fill field.";
    if (param.action(i).has_copy()) {
      CHECK_GE(param.action(i).copy(), 0) << "action.copy must be non-negative";
      CHECK_LT(param.action(i).copy(), bottom[0]->count(1)) << "action.copy must be less than number of channels in bottom[0]";
    }
  }
}

template <typename Dtype>
void ChannelPermutationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2) << "Bottom blob must be at least 2-dimensional.";
  const ChannelPermutationParameter& param = this->layer_param_.channel_permutation_param();

  inplace = bottom[0]==top[0] && param.inplace_possible() && bottom[0]->shape(0)==1;
  use_temp_storage = !this->inplace && bottom[0]==top[0];
  really_inplace = inplace && !use_temp_storage;

  // Reshape the first blob and temporary storage.
  vector<int> top_shape = bottom[0]->shape();
  CHECK_GE(top_shape.size(), 2);
  top_shape[1] = param.num_output();
  top[0]->Reshape(top_shape);
  if (use_temp_storage) this->temp.Reshape(top_shape);
}

template <typename Dtype>
void ChannelPermutationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ChannelPermutationParameter& param = this->layer_param_.channel_permutation_param();
  const bool cpu_mode = Caffe::mode() == Caffe::CPU;

  const Dtype* bottom_data = cpu_mode ? bottom[0]->cpu_data() : bottom[0]->gpu_data();
  Dtype* top_data = cpu_mode ? top[0]->mutable_cpu_data() : top[0]->mutable_gpu_data();
  Dtype* dst_data = use_temp_storage ? (cpu_mode ? this->temp.mutable_cpu_data() : this->temp.mutable_gpu_data()) : top_data;

  const int count = top[0]->count();
  const int num_samples = top[0]->shape(0);
  const int num_channels = top[0]->shape(1);
  CHECK_EQ(num_channels, param.num_output());
  const int num_per_sample_old = bottom[0]->count(1);
  const int num_per_sample_new = top[0]->count(1);
  const int num_per_channel = bottom[0]->count(2);

  void (*caffe_set_impl)(const int N, const Dtype alpha, Dtype* Y) =
#ifndef CPU_ONLY
      cpu_mode ? caffe_set<Dtype> : caffe_gpu_set<Dtype>;
#else
      caffe_set<Dtype>;
#endif

  int sample_offset_old = 0;
  int sample_offset_new = 0;
  for (int n = 0; n < num_samples; ++n, sample_offset_old+=num_per_sample_old, sample_offset_new+=num_per_sample_new) {
    DCHECK_LE(sample_offset_new+num_per_sample_new, count);
    DCHECK_EQ(sample_offset_old, n * num_per_sample_old);
    DCHECK_EQ(sample_offset_new, n * num_per_sample_new);

    int next_chan = 0;
    int next_chan_offset = 0;

    for (int i = 0; i < param.action_size(); ++i) {
      DCHECK_EQ(next_chan_offset, next_chan * num_per_channel);

      int cur_chan = param.action(i).chan();
      if (!really_inplace) {  // process kept channels up to cur_chan
        if (next_chan<cur_chan) {
          CHECK_LE(sample_offset_new + next_chan_offset + num_per_channel * (cur_chan - next_chan), count) << "1";
          caffe_copy(num_per_channel * (cur_chan - next_chan),
              bottom_data + sample_offset_old + next_chan_offset,
              dst_data + sample_offset_new + next_chan_offset);
          next_chan = cur_chan;
          next_chan_offset = next_chan * num_per_channel;
        }
      } else {
        next_chan = cur_chan;
        next_chan_offset = next_chan * num_per_channel;
      }

      DCHECK_LT(sample_offset_new + next_chan_offset, count) << "2_" << int(really_inplace);
      if (param.action(i).has_copy()) {
        int src_chan = param.action(i).copy();
        caffe_copy(num_per_channel,
            bottom_data + sample_offset_old + src_chan * num_per_channel,
            dst_data + sample_offset_new + next_chan_offset);
      } else {   // here: param.action(i).has_fill()==true -- checked in LayerSetUp
        DCHECK(param.action(i).has_copy() || param.action(i).has_fill()) << "ChannelPermutationAction must have either copy or fill field.";
        caffe_set_impl(num_per_channel,
            param.action(i).fill(),
            dst_data + sample_offset_new + next_chan_offset);
      }
      ++next_chan;
      next_chan_offset += num_per_channel;
    }

    DCHECK_EQ(next_chan_offset, next_chan * num_per_channel);
    if (!really_inplace) {  // process kept channels up to the last
      const int cur_chan = num_channels;
      if (next_chan<cur_chan) {
        DCHECK_LE(sample_offset_new + next_chan_offset + num_per_channel * (cur_chan - next_chan), count) << "3";
        caffe_copy(num_per_channel * (cur_chan - next_chan),
            bottom_data + sample_offset_old + next_chan_offset,
            dst_data + sample_offset_new + next_chan_offset);
      }
    }
  }
  if (use_temp_storage)  caffe_copy(top[0]->count(), dst_data, top_data);
}

template <typename Dtype>
void ChannelPermutationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])  return;

  CHECK(false) << "Backward pass not implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(ChannelPermutationLayer);
#endif

INSTANTIATE_CLASS(ChannelPermutationLayer);
REGISTER_LAYER_CLASS(ChannelPermutation);

}  // namespace caffe
