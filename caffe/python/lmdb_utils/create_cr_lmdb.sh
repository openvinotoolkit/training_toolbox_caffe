cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="/data"
dataset_name="crossroad_100"
mapfile="$root_dir/python/lmdb_utils/labelmap_cr.prototxt"
anno_type="detection"
label_type="xml"
db="lmdb"
min_dim=0
max_dim=0
width=1024
height=1024

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
  extra_cmd="$extra_cmd --redo"
fi

subset="train"

PYTHONPATH=$PYTHONPATH:$root_dir/python python3 $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height  --check-label $extra_cmd --shuffle --backend=$db $data_root_dir $data_root_dir/$subset.txt $data_root_dir/$db/$dataset_name"_"$subset"_"$db /tmp/tmp 1>&1 | tee $root_dir/python/lmdb_utils/$subset.log

