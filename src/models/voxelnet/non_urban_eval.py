import glob
import argparse
import os
import time
import tensorflow as tf
from model import RPN3D
from config import cfg
from utils import *
from utils.kitti_loader import iterate_data, sample_test_data

ctx = 'non_urban' 
tag = 'pre_trained_car'
output_path = 'default_non_urban' 
exp = 'non_urban'
bs=4 
vis=True
GPU_AVAILABLE='0'
GPU_USE_COUNT=1

dataset_dir = cfg.DATA_DIR
val_dir = os.path.join(cfg.CONTEXT_DIR, ctx)
save_model_dir = os.path.join('./save_model', tag)

# experiment = Experiment(api_key="xXtJguCo8yFdU7dpjEpo6YbHw",project_name=exp)
# create output folder
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'data'), exist_ok=True)
if vis:
    os.makedirs(os.path.join(output_path, 'vis'), exist_ok=True)


with tf.Graph().as_default():

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                        visible_device_list=GPU_AVAILABLE,
                        allow_growth=True)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        device_count={
            "GPU": GPU_USE_COUNT,
        },
        allow_soft_placement=True,
        log_device_placement=True
    )

    with tf.Session(config=config) as sess:
        model = RPN3D(
            cls=cfg.DETECT_OBJ,
            single_batch_size=bs,
            avail_gpus=GPU_AVAILABLE
        )
        if tf.train.get_checkpoint_state(save_model_dir):
            print("Reading model parameters from %s" % save_model_dir)
            model.saver.restore(
                sess, tf.train.latest_checkpoint(save_model_dir))
        counter=0
#         with experiment.test():
        for batch in iterate_data(val_dir, shuffle=False, aug=False, is_testset=False, batch_size=bs * GPU_USE_COUNT, multi_gpu_sum=GPU_USE_COUNT):
#             experiment.log_metric("counter",counter)

            if vis:
                tags, results, front_images, bird_views, heatmaps = model.predict_step(sess, batch, summary=False, vis=True)
            else:
                tags, results = model.predict_step(sess, batch, summary=False, vis=False)

            # ret: A, B
            # A: (N) tag
            # B: (N, N') (class, x, y, z, h, w, l, rz, score)
            for tag, result in zip(tags, results):
                of_path = os.path.join(output_path, 'data', tag + '.txt')
                with open(of_path, 'w+') as f:
                    labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                    for line in labels:
                        f.write(line)
                    print('write out {} objects to {}'.format(len(labels), tag))
            # dump visualizations
            if vis:
                for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps):
                    front_img_path = os.path.join( output_path, 'vis', tag + '_front.jpg'  )
                    bird_view_path = os.path.join( output_path, 'vis', tag + '_bv.jpg'  )
                    heatmap_path = os.path.join( output_path, 'vis', tag + '_heatmap.jpg'  )
                    cv2.imwrite( front_img_path, front_image )
                    cv2.imwrite( bird_view_path, bird_view )
                    cv2.imwrite( heatmap_path, heatmap )
            counter +=1
