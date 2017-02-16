#include "caffe/layers/ctc_decoder_layer.hpp"

#include <algorithm>
#include <vector>

// Base decoder
// ============================================================================

namespace caffe {
template <typename Dtype>
CTCDecoderLayer<Dtype>::CTCDecoderLayer(const LayerParameter& param)
  : Layer<Dtype>(param)
  , T_(0)
  , N_(0)
  , C_(0)
  , blank_index_(param.ctc_decoder_param().blank_index())
  , merge_repeated_(param.ctc_decoder_param().ctc_merge_repeated())
  , sequence_index_(0)
  , score_index_(-1)
  , accuracy_index_(-1) {
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // compute indices of output (top) blobs
  sequence_index_ = 0;  // always
  if (bottom.size() == 2) {
    // 2 input blobs (data, sequence indicators)

    if (top.size() == 1) {
      // no further output
    } else if (top.size() == 2) {
      score_index_ = 1;  // output scores
    } else {
      LOG(FATAL) << "Only two output blobs allowed: "
                 << "1: sequences, 2: scores";
    }
  } else if (bottom.size() == 3) {
    // 3 input blobs (data, seq_ind, target_seq)
    if (top.size() == 1) {
      // no further output
    } else if (top.size() == 2) {
      accuracy_index_ = 1;  // output accuracy
    } else if (top.size() == 3) {
      score_index_ = 1;  // output scores
      accuracy_index_ = 2;  // output accuracy
    } else {
      LOG(FATAL) << "Need two or three output blobs: "
                 << "a) 1: sequences, 2: accuracy, or "
                 << "b) 1: sequences, 2: score, 3: accuracy.";
    }
  }
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>* probabilities = bottom[0];
  T_ = probabilities->shape(0);
  N_ = probabilities->shape(1);
  C_ = probabilities->shape(2);

  output_sequences_.clear();
  output_sequences_.resize(N_);

  if (sequence_index_ >= 0) {
    Blob<Dtype>* sequences = top[sequence_index_];
    sequences->Reshape(N_, T_, 1, 1);  // switch to N x T
  }

  if (score_index_ >= 0) {
    Blob<Dtype>* scores = top[score_index_];
    scores->Reshape(N_, 1, 1, 1);
  }

  if (accuracy_index_ >= 0) {
    Blob<Dtype>* accuracy = top[accuracy_index_];
    accuracy->Reshape(1, 1, 1, 1);
  }

  if (blank_index_ < 0) {
    blank_index_ = C_ - 1;
  }
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>* probabilities = bottom[0];
  const Blob<Dtype>* sequence_indicators = bottom[1];
  Blob<Dtype>* scores = (score_index_ >= 0) ? top[score_index_] : 0;

  // decode string with the requiested method (e.g. CTCGreedyDecoder)
  Decode(probabilities, sequence_indicators, &output_sequences_, scores);

  // transform output_sequences to blob
  if (sequence_index_ >= 0) {
    Blob<Dtype>* sequence_blob = top[sequence_index_];
    Dtype* sequence_d = sequence_blob->mutable_cpu_data();
    // clear all data
    caffe_set(sequence_blob->count(), static_cast<Dtype>(-1), sequence_d);

    // copy data
    for (int n = 0; n < N_; ++n) {
      Dtype* seq_n_d = sequence_d + sequence_blob->offset(n, 0);
      const Sequence &output_seq = output_sequences_[n];
      CHECK_LE(output_seq.size(), T_);
      for (size_t t = 0; t < output_seq.size(); ++t) {
        seq_n_d[t] = output_seq[t];
      }
    }
  }

  // compute accuracy
  if (accuracy_index_ >= 0) {
    Dtype &acc = top[accuracy_index_]->mutable_cpu_data()[0];
    acc = 0;

    CHECK_GE(bottom.size(), 3);  // required target sequences blob
    const Blob<Dtype>* target_sequences_data = bottom[2];
    const Dtype* ts_data = target_sequences_data->cpu_data();
    for (int n = 0; n < N_; ++n) {

      Sequence target_sequence;
      for (int t = 0; t < T_; ++t) {
        const Dtype dtarget = ts_data[target_sequences_data->offset(t, n)];
        if (dtarget < 0) {
          // sequence has finished
          break;
        }
        // round to int, just to be sure
        const int target = static_cast<int>(0.5 + dtarget);
        target_sequence.push_back(target);
      }
      if (std::equal(target_sequence.begin(), target_sequence.end(), output_sequences_[n].begin()))
          ++acc;
      if (std::max(target_sequence.size(), output_sequences_[n].size()) == 0) {
        // 0 length
        continue;
      }

//      const int ed = EditDistance(target_sequence, output_sequences_[n]);

//      acc += ed * 1.0 /
//              std::max(target_sequence.size(), output_sequences_[n].size());
    }

    //acc = 1 - acc / N_;
    acc = acc / N_;
    CHECK_GE(acc, 0);
    CHECK_LE(acc, 1);
  }
}

template <typename Dtype>
void CTCDecoderLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { NOT_IMPLEMENTED; }
  }
}

template <typename Dtype>
int CTCDecoderLayer<Dtype>::EditDistance(const Sequence &s1,
                                         const Sequence &s2) {
  const size_t len1 = s1.size();
  const size_t len2 = s2.size();

  Sequences d(len1 + 1, Sequence(len2 + 1));

  d[0][0] = 0;
  for (size_t i = 1; i <= len1; ++i) {d[i][0] = i;}
  for (size_t i = 1; i <= len2; ++i) {d[0][i] = i;}

  for (size_t i = 1; i <= len1; ++i) {
    for (size_t j = 1; j <= len2; ++j) {
      d[i][j] = std::min(
                  std::min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1),
                  d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1));
    }
  }

  return d[len1][len2];
}

INSTANTIATE_CLASS(CTCDecoderLayer);


// Greedy decoder
// ============================================================================

template <typename Dtype>
void CTCGreedyDecoderLayer<Dtype>::Decode(
        const Blob<Dtype>* probabilities,
        const Blob<Dtype>* sequence_indicators,
        Sequences* output_sequences,
        Blob<Dtype>* scores) const {
  Dtype* score_data = 0;
  if (scores) {
    CHECK_EQ(scores->count(), N_);
    score_data = scores->mutable_cpu_data();
    caffe_set(N_, static_cast<Dtype>(0), score_data);
  }

  for (int n = 0; n < N_; ++n) {
    int prev_class_idx = -1;

    for (int t = 0; /* check at end */; ++t) {
      // get maximum probability and its index
      int max_class_idx = 0;
      const Dtype* probs = probabilities->cpu_data()
              + probabilities->offset(t, n);
      Dtype max_prob = probs[0];
      ++probs;
      for (int c = 1; c < C_; ++c, ++probs) {
          if (*probs > max_prob) {
              max_class_idx = c;
              max_prob = *probs;
          }
      }

      if (score_data) {
        score_data[n] += -max_prob;
      }

      if (max_class_idx != blank_index_
              && !(merge_repeated_&& max_class_idx == prev_class_idx)) {
          output_sequences->at(n).push_back(max_class_idx);
      }

      prev_class_idx = max_class_idx;

      if (t + 1 == T_ || sequence_indicators->data_at(t + 1, n, 0, 0) == 0) {
          break;
      }
    }
  }
}

INSTANTIATE_CLASS(CTCGreedyDecoderLayer);
REGISTER_LAYER_CLASS(CTCGreedyDecoder);

// Beam search decoder implementation based on prefix search decoder, see:
// Graves, A. Supervised Sequence Labelling with Recurrent Neural Networks, 2012
// ============================================================================

template <typename Dtype>
Dtype LogPSum(Dtype a, Dtype b) {
  return a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b));
}

template <typename Dtype>
void CTCBeamSearchDecoderLayer<Dtype>::Decode(
        const Blob<Dtype>* log_probabilities,
        const Blob<Dtype>* sequence_indicators,
        Sequences* output_sequences,
        Blob<Dtype>* scores) const {
  Dtype* score_data = 0;
  if (scores) {
    CHECK_EQ(scores->count(), N_);
    score_data = scores->mutable_cpu_data();
    caffe_set(N_, static_cast<Dtype>(0), score_data);
  }

  for (int n = 0; n < N_; ++n) {
    Prefixes to_expand;
    std::vector<Candidate> paths;

    // empty root prefix
    Candidate root;
    root.label = blank_index_;
    root.parent = -1;
    root.expanded = false;
    root.lPn = -INFINITY;
    root.lPt = root.lPb = 0.;
    paths.push_back(root);

    to_expand.insert(Node(0., 0));

    for (int t = 0; t < T_; t++) {
      int fill_from = paths.size();
      const Dtype* logp = log_probabilities->cpu_data() +
          log_probabilities->offset(t, n);

      // expand all candidates
      for (typename Prefixes::reverse_iterator it = to_expand.rbegin();
           it != to_expand.rend(); it++) {
        int parent = it->second;

        // build all children
        for (int k = 0; k < C_; ++k) {
          if (k == blank_index_)
            continue;
          Candidate e;
          e.label = k;
          e.parent = parent;
          e.expanded = false;
          if (paths[parent].label == k)
            e.lPn = paths[parent].lPb + logp[k];
          else
            e.lPn = paths[parent].lPt + logp[k];
          e.lPb = -INFINITY;
          e.lPt = e.lPn;

          paths.push_back(e);
        }

        paths[parent].expanded = true;
      }

      // update logprobs
      for (int p = fill_from - 1; p >= 0; p--) {
        int parent = paths[p].parent;
        if (parent != -1) {
          if (paths[parent].label == paths[p].label)
            paths[p].lPn = LogPSum(paths[p].lPn, paths[parent].lPb);
          else
            paths[p].lPn = LogPSum(paths[p].lPn, paths[parent].lPt);
          paths[p].lPn += logp[paths[p].label];
        }
        paths[p].lPb = paths[p].lPt + logp[blank_index_];
        paths[p].lPt = LogPSum(paths[p].lPn, paths[p].lPb);
      }

      if (t + 1 == T_ || sequence_indicators->data_at(t + 1, n, 0, 0) == 0) {
          break;
      }

      // fill new candidates to expand
      to_expand.clear();
      for (int p = 0; p < paths.size(); p++) {
        if (! paths[p].expanded && paths[p].lPt > -INFINITY) {
          to_expand.insert(Node(paths[p].lPt, p));
          if (to_expand.size() > max_candidates_)
            to_expand.erase(to_expand.begin());
        }
      }
    }

    // get the label(s)
    int top_n_ = 1;  // TODO: make it param

    // fill all candidates to labels
    to_expand.clear();
    for (int p = 0; p < paths.size(); p++) {
      to_expand.insert(Node(paths[p].lPt, p));
      if (to_expand.size() > top_n_)
        to_expand.erase(to_expand.begin());
    }

    for (typename Prefixes::reverse_iterator it = to_expand.rbegin();
         it != to_expand.rend(); it++) {
      Dtype label_lpt = it->first;
      int label = it->second;

      Sequence l;
      while (label > 0) {
        l.push_back(paths[label].label);
        label = paths[label].parent;
      }

      // TODO: multi labels output
      std::reverse(l.begin(), l.end());
      (*output_sequences)[n] = l;
      if (score_data) {
        score_data[n] = -label_lpt;  // negated logprob
      }
    }
  }
}

INSTANTIATE_CLASS(CTCBeamSearchDecoderLayer);
REGISTER_LAYER_CLASS(CTCBeamSearchDecoder);

}  // namespace caffe
