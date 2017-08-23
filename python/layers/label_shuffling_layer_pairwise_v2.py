import yaml
import caffe
from caffe.io import caffe_pb2
import numpy as np
import lmdb
from tqdm import tqdm
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict, Counter

def parseRecord(val):
    datum = caffe_pb2.Datum.FromString(val)
    data = caffe.io.datum_to_array(datum)
    label = np.array([datum.label])

    return data, label

def initialSetup(path):
    keys_labels = dict()
    labels_keys = dict()
    env = lmdb.open(path, readonly=True)
    with env.begin() as txn:
        print env.stat()
        cursor = txn.cursor()
        for k, value in tqdm(cursor, desc='label_shuffling_layer initialSetup', total=env.stat()['entries']):
            datum = caffe_pb2.Datum.FromString(value)
            arr = caffe.io.datum_to_array(datum)
            label = datum.label;
            if not labels_keys.has_key(label):
                labels_keys[label] = []
            labels_keys[label].append(k)
            keys_labels[k] = label

    return keys_labels, labels_keys

# generate batch with predefined number of different classes and images per class
class LabelShufflingLayerPairwise(caffe.Layer):
    def setup(self, bottom, top):
        param_str = self.param_str.replace(";;", "\n") #for debugging only
        layer_params = yaml.load(param_str)

        self.source_ = layer_params['source']
        print self.source_
        self.scales_ = layer_params['scales']
        self.subtract_ = np.zeros_like(self.scales_)
        if 'subtract' in layer_params:
            self.subtract_ = layer_params['subtract']

        self.batch_size_ = layer_params['batch_size']  # self.num_labels_*self.images_per_label_

        self.max_number_object_per_label_ = np.inf
        if 'max_number_object_per_label' in layer_params:
            self.max_number_object_per_label_ = layer_params['max_number_object_per_label']

        self.min_labels_to_choose_chunk_ = 4
        if 'min_labels_to_choose_chunk' in layer_params:
            self.min_labels_to_choose_chunk_ = layer_params['min_labels_to_choose_chunk']

        self.mirror_ = False
        if 'mirror' in layer_params:
            self.mirror_ = layer_params['mirror']

        self.dither_ = False
        if 'dither' in layer_params:
            self.dither_ = layer_params['dither']

        self.blur_ = False
        if 'blur' in layer_params:
            self.blur_ = layer_params['blur']

        self.change_brightness_ = False
        if 'change_brightness' in layer_params:
            self.change_brightness_ = layer_params['change_brightness']

        self.min_share_of_positives_in_chunk_ = 0.4
        if 'min_share_of_positives_in_chunk' in layer_params:
            self.min_share_of_positives_in_chunk_ = layer_params['min_share_of_positives_in_chunk']

        self.min_share_of_negatives_in_chunk_ = 0.4
        if 'min_share_of_negatives_in_chunk' in layer_params:
            self.min_share_of_negatives_in_chunk_ = layer_params['min_share_of_negatives_in_chunk']

        assert(self.min_share_of_negatives_in_chunk_ + self.min_share_of_positives_in_chunk_ <= 1.0)

        self.max_num_attempts_to_choose_pair_ = 10
        self.should_debug_ = bool(int(layer_params.get('debug', 0)))

        # structures setup
        self.keys_labels_, self.labels_keys_ = initialSetup(self.source_)
        self.keys_ = self.keys_labels_.keys()
        self.labels_ = self.labels_keys_.keys()

        np.random.shuffle(self.labels_)

        self.label_index_ = 0
        self.image_index_ = 0

        # figure out the shape
        k = self.keys_[0]

        env = lmdb.open(self.source_, readonly=True)
        with env.begin() as txn:
            val = txn.get(k)
        data, label = parseRecord(val)
        self.data_shape_ = data.shape
        self.label_shape_ = label.shape

        self.chunk_size_ = layer_params.get('chunk_size', self.batch_size_)

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size_, self.data_shape_[2], self.data_shape_[0], self.data_shape_[1])
        top[1].reshape(self.batch_size_, self.data_shape_[2], self.data_shape_[0], self.data_shape_[1])
        top[2].reshape(self.batch_size_, self.label_shape_[0])

    def getNewLabel(self, labels_not_choose, labels_list):
        if set(labels_not_choose) > set(labels_list):
            return None

        while True:
            new_label = np.random.choice(labels_list)
            if not new_label in labels_not_choose:
                return new_label

    def getNewKey(self, label, old_keys):
        while True:
            new_key = np.random.choice(self.labels_keys_[label])

            if not new_key in old_keys:
                return new_key

    def getBatchChunk(self, chunk_size=None):
        def choose_batch_pair_with_same_labels(self, chosen_key_pairs, chosen_labels, num_chosen_pairs_for_label, set_labels_been_chosen):
                        cur_label = np.random.choice(chosen_labels)
                        cur_keys = self.labels_keys_[cur_label]
                        for num_attempt in xrange(self.max_num_attempts_to_choose_pair_):
                            key1 = np.random.choice(cur_keys)
                            key2 = np.random.choice(cur_keys)
                            if (key1 == key2) or ((key1, key2) in chosen_key_pairs) or ((key2, key1) in chosen_key_pairs):
                                continue
                            chosen_key_pairs.add( (key1, key2) )
                            num_chosen_pairs_for_label[cur_label] += 2
                            self.print_dbg("choose_batch_pair_with_same_labels: use label '{}'".format(cur_label))
                            return True

                        chosen_labels.remove(cur_label) #remove from list but not from set_labels_been_chosen
                        new_label = self.getNewLabel(set_labels_been_chosen, self.labels_)
                        if new_label is None:
                            raise RuntimeError("All labels have been looked through, dataset is too small")
                        chosen_labels.append(new_label)
                        set_labels_been_chosen.add(new_label)
                        self.print_dbg("choose_batch_pair_with_same_labels: remove label '{}', add label '{}'".format(cur_label, new_label))
                        return False

        def choose_batch_pair_with_diff_labels(self, chosen_key_pairs, chosen_labels, num_chosen_pairs_for_label, set_labels_been_chosen):
                        cur_label1 = np.random.choice(chosen_labels)
                        cur_label2 = self.getNewLabel([cur_label1], chosen_labels)
                        assert(cur_label1 != cur_label2)
                        cur_keys1 = self.labels_keys_[cur_label1]
                        cur_keys2 = self.labels_keys_[cur_label2]
                        for num_attempt in xrange(self.max_num_attempts_to_choose_pair_):
                            key1 = np.random.choice(cur_keys1)
                            key2 = np.random.choice(cur_keys2)
                            if (key1 == key2) or ((key1, key2) in chosen_key_pairs) or ((key2, key1) in chosen_key_pairs):
                                continue
                            chosen_key_pairs.add( (key1, key2) )
                            num_chosen_pairs_for_label[cur_label1] += 1
                            num_chosen_pairs_for_label[cur_label2] += 1
                            self.print_dbg("choose_batch_pair_with_diff_labels: use labels '{}' and '{}'".format(cur_label1, cur_label2))
                            return True

                        #TODO: think if we should use relative or absolute values to choose which label should be removed
                        val1 = num_chosen_pairs_for_label[cur_label1]
                        val2 = num_chosen_pairs_for_label[cur_label2]
                        if val1 > val2:
                            label_to_remove = cur_label1
                        else:
                            label_to_remove = cur_label2
                        chosen_labels.remove(label_to_remove) #remove from list but not from set_labels_been_chosen
                        new_label = self.getNewLabel(set_labels_been_chosen, self.labels_)
                        if new_label is None:
                            raise RuntimeError("All labels have been looked through, dataset is too small")
                        chosen_labels.append(new_label)
                        set_labels_been_chosen.add(new_label)
                        self.print_dbg("choose_batch_pair_with_diff_labels: cannot choose for labels ('{}', '{}'), remove label '{}', add label '{}'".format(cur_label1, cur_label2, label_to_remove, new_label))
                        return False

        def choose_batch_pair(self, should_be_same, chosen_key_pairs, chosen_labels, num_chosen_pairs_for_label, set_labels_been_chosen):
                        if should_be_same:
                            return choose_batch_pair_with_same_labels(self, chosen_key_pairs, chosen_labels, num_chosen_pairs_for_label, set_labels_been_chosen)
                        else:
                            return choose_batch_pair_with_diff_labels(self, chosen_key_pairs, chosen_labels, num_chosen_pairs_for_label, set_labels_been_chosen)

        if chunk_size == None:
            chunk_size = self.batch_size_

        chosen_key_pairs = set()
        num_chosen_pairs_for_label = defaultdict(int)

        num_chosen_labels = self.min_labels_to_choose_chunk_
        chosen_labels = np.random.permutation(self.labels_)[:num_chosen_labels].tolist()
        num_chosen_labels = len(chosen_labels)
        set_labels_been_chosen = set(chosen_labels)
        assert(len(set_labels_been_chosen) < len(self.labels_))

        min_prob_same = self.min_share_of_positives_in_chunk_
        max_prob_same = 1.0 - self.min_share_of_negatives_in_chunk_
        assert(max_prob_same >= min_prob_same)
        prob_same_for_chunk = np.random.random() * (max_prob_same - min_prob_same) + min_prob_same

        while len(chosen_key_pairs) < chunk_size:
            assert(len(set_labels_been_chosen) < len(self.labels_))
            should_be_same = np.random.random() > prob_same_for_chunk
            ok = False
            while not ok:
                ok = choose_batch_pair(self, should_be_same, chosen_key_pairs, chosen_labels, num_chosen_pairs_for_label, set_labels_been_chosen)

        batch_keypairs_structs = []
        for k1, k2 in chosen_key_pairs:
            keypair_struct = {  "key1": k1,
                                "key2": k2,
                                "are_same": self.keys_labels_[k1]==self.keys_labels_[k2]}
            batch_keypairs_structs.append(keypair_struct)

        self.print_dbg("getBatchChunk: num_chosen_labels = {}, len(set_labels_been_chosen) = {}, chunk_size={}".format(num_chosen_labels, len(set_labels_been_chosen), chunk_size))
        return batch_keypairs_structs

    def get_image_pairs(self, n):
        imgs_left = []
        imgs_right = []

        labels_left_all = []
        labels_right_all = []

        pair_label = []

        current_batch_size = 0
        while current_batch_size < n:
            labels_left = []
            labels_right = []

            current_chunk_size  = min(self.chunk_size_, n - current_batch_size)
            batch_keypairs_structs = self.getBatchChunk(current_chunk_size)

            env = lmdb.open(self.source_, readonly=True)

            with env.begin() as txn:
                for keypair_struct in batch_keypairs_structs:
                    k1 = keypair_struct["key1"]
                    k2 = keypair_struct["key2"]
                    same_label = keypair_struct["are_same"]

                    img1, label1 = self.getAugmentedImageWithLabelFromDb(txn, k1)
                    img2, label2 = self.getAugmentedImageWithLabelFromDb(txn, k2)
                    assert(label1 == self.keys_labels_[k1])
                    assert(label2 == self.keys_labels_[k2])
                    assert(same_label == (int(label1) == int(label2)))

                    imgs_left.append(img1)
                    labels_left.append(label1)

                    imgs_right.append(img2)
                    labels_right.append(label2)

                    pair_label.append([1] if same_label else [0])

            current_batch_size += current_chunk_size

            labels_left_all += labels_left
            labels_right_all += labels_right

        sum_pair_label_vals = sum([v for one_el_list in pair_label for v in one_el_list])
        percents_same_label = round(100 * float(sum_pair_label_vals) / len(pair_label))
        print "The generated batch contains {}% pairs with same labels and {}% pairs with different labels".format(percents_same_label, 100-percents_same_label)

        return imgs_left, imgs_right, pair_label

    def getAugmentedImageWithLabelFromDb(self, txn, key):
        val = txn.get(key)
        fields = parseRecord(val)

        img = self.convertToCaffeLayout(self.augment(fields[0]))
        label = fields[1]
        return img, label

    def convertToCaffeLayout(self, img):
        img = img.transpose((2, 0, 1))
        return img # w, h, c -> c, w, h

    def augment(self, img):
        augmented_img = img

        if self.blur_:
            rand = np.random.randint(0, 2)
            if rand == 1:
                filter_size = np.random.uniform(low=0.0, high=0.5)

                for i in range(3):
                    augmented_img[:, :, i] = gaussian_filter(augmented_img[:, :, i], sigma=filter_size)

        if self.dither_:
            width = self.data_shape_[1]
            height = self.data_shape_[0]

            min_factor = 0
            max_factor_left_right = 0.05
            max_factor_top = 0.05
            max_factor_bottom = 0.05

            distance_from_edge_left = int(np.random.uniform(min_factor, max_factor_left_right) * width)
            right_edge = int(width * (1 - np.random.uniform(min_factor, max_factor_left_right)))
            distance_from_edge_top = int(np.random.uniform(min_factor, max_factor_top) * height)
            bottom_edge = int(height * (1 - np.random.uniform(min_factor, max_factor_bottom)))

            crop = augmented_img[distance_from_edge_top : bottom_edge, distance_from_edge_left : right_edge, ::]
            augmented_img = imresize(crop, (height, width))

        if self.mirror_:
            rand = np.random.randint(0, 2)
            if rand == 1:
                augmented_img = augmented_img[:, ::-1, :]

        if self.change_brightness_:
            rand = np.random.randint(0, 2)
            if rand == 1:
                alpha = np.random.uniform(0.5, 1.5)
                beta = np.random.randint(-50, 50)
                changed_brightness = augmented_img * alpha + beta

                augmented_img = np.where(changed_brightness < 255,
                                         changed_brightness,
                                         np.full_like(augmented_img, 255, dtype=np.uint8))
                augmented_img = np.where(augmented_img >= 0,
                                         augmented_img,
                                         np.full_like(augmented_img, 0, dtype=np.uint8))

        return augmented_img

    def forward(self, bottom, top):
        imgs_left, imgs_right, pair_label = self.get_image_pairs(self.batch_size_)

        top[0].data[...] = (np.array(imgs_left) - self.subtract_[0]) * self.scales_[0]
        top[1].data[...] = (np.array(imgs_right) - self.subtract_[0]) * self.scales_[0]
        top[2].data[...] = np.array(pair_label)

    def backward(self, top, propagate_down, bottom):
        pass

    def print_dbg(self, *args):
        if self.should_debug_:
            print " ".join([str(x) for x in args])
