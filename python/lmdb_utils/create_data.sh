cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="path_to_WIDERFace_dataset"
dataset_name="wider"
mapfile="$root_dir/python/lmdb_utils/labelmap_wider.prototxt"
anno_type="detection"
label_type="xml"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in wider_train wider_val
do
  PYTHONPATH=$PYTHONPATH:$root_dir/python python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/$subset.txt $data_root_dir/$db/$dataset_name"_"$subset"_"$db $data_root_dir/examples/$dataset_name 2>&1 | tee $root_dir/python/lmdb_utils/$subset.log
done
