/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_

#ifdef USE_NCCL

#include <boost/thread.hpp>

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/nccl.hpp"

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype>
class Params {
 public:
  explicit Params(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~Params() {
  }

  inline size_t size() const {
    return size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }

 protected:
  const size_t size_;           // Size of buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient

DISABLE_COPY_AND_ASSIGN(Params);
};

// Params stored in GPU memory.
template<typename Dtype>
class GPUParams : public Params<Dtype> {
 public:
  GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device);
  virtual ~GPUParams();

  void Configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

template<typename Dtype>
class NCCL : public GPUParams<Dtype>,
             public Solver<Dtype>::Callback,
             public Net<Dtype>::Callback {
 public:
  /**
   * Single process version.
   */
  explicit NCCL(shared_ptr<Solver<Dtype> > solver);
  /**
   * In multi-process settings, first create a NCCL id (new_uid), then
   * pass it to each process to create connected instances.
   */
  NCCL(shared_ptr<Solver<Dtype> > solver, const string& uid);
  ~NCCL();

  boost::barrier* barrier();
  void set_barrier(boost::barrier* value);

  /**
   * In single process settings, create instances without uids and
   * call this to connect them.
   */
  static void InitSingleProcess(vector<NCCL<Dtype>*>* nccls);

  static string new_uid();

  /**
   * Broadcast weights from rank 0 other solvers.
   */
  void Broadcast();

  /**
   * Single process multi-GPU.
   */
  void Run(const vector<int>& gpus, const char* restore);

 protected:
  void Init();
  void on_start() {}
  void run(int layer);  // Net callback
  void on_gradients_ready();

  ncclComm_t comm_;
  cudaStream_t stream_;

  shared_ptr<Solver<Dtype> > solver_;
  // Should not be necessary, https://github.com/NVIDIA/nccl/issues/37
  boost::barrier* barrier_;
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif  // USE_NCCL
#endif  // header
