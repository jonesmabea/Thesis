# crooped gt dir
GT_DIR=/mnt/storage/home/ja17618/scratch/DATA_DIR/validation/label_2
# pred dir
PRED_DIR=$1

# output log
OUTPUT=$2
# start test
nohup `pwd`/kitti_eval/evaluate_object_3d_offline $GT_DIR $PRED_DIR > $OUTPUT 2>&1 &
