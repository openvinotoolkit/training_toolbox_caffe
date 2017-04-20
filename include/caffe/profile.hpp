#ifndef CAFFE_PROFILE_HPP_
#define CAFFE_PROFILE_HPP_

#include <vector>

namespace caffe {

struct ProfilingStat {
  ProfilingStat()
      : forward_time(0.0),
        backward_time(0.0),
        forward_iteration(0),
        backward_iteration(0) {
  }
  std::vector<double> forward_time_per_layer;
  std::vector<double> backward_time_per_layer;
  double forward_time;
  double backward_time;
  std::size_t forward_iteration;
  std::size_t backward_iteration;
};

}  // namespace caffe

#endif  // CAFFE_PROFILE_HPP_
