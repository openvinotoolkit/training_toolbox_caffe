
import sys
import os
os.environ['GLOG_log_dir'] = "/tmp/"
import caffe
from label_shuffling_layer_pairwise_v2 import LabelShufflingLayerPairwise

import tempfile

def simple_net_file():
    """Make a simple net prototxt, based on test_net.cpp, returning the name
    of the (temporary) file."""

    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""name: 'testnet' force_backward: true
layer {
  type: 'Python'
  name: 'data'
  top: 'data'
  top: 'data_right'
  top: 'pair_label'
  python_param {
    module: 'label_shuffling_layer_pairwise_v2'
    layer: 'LabelShufflingLayerPairwise'
    param_str:  "source: /home/lbeynens/_Work/_Datasets/Faces/LMDBs/part0.90__merged__internet_crawling__morph0.1__western_celebs_.copy//;;batch_size: 290;;chunk_size: 40;;scales : [0.00390625, 1.];;max_number_object_per_label : 50000;;same_label_in_batch : 30;;mirror : False;;dither : False;;blur: False;;change_brightness: False;;debug: 1"
  }
}
""")
    f.close()
    return f.name

net_file = simple_net_file()
net = caffe.Net(net_file, caffe.TRAIN)
os.remove(net_file)

for i in xrange(5):
    print ("=" * 80)
    net.forward()
