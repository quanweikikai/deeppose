#! /bin/bash
caffe_dir=$HOME/caffe

if [ ! -f baseball/data/image_mean.binaryproto ]; then
    $caffe_dir/build/tools/compute_image_mean baseball/image_train1.lmdb baseball/image_mean1.binaryproto
fi

#caffe train   -solver=/home/xacti-dnn1/HDD/DNN/deeppose/models/AlexNet/baseball_solver1.prototxt -gpu=1
caffe train   -solver=/home/xacti-dnn1/HDD/DNN/deeppose/models/AlexNet/baseball_solver.prototxt -gpu=1
