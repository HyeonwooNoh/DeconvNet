##########################
# start stage 2 training #
##########################
cd 002_stage_2

## create simulinks

# image data
ln -s ../../data/VOC2012
# training / validataion imageset
ln -s ../../data/imagesets/stage_2_train_imgset
# caffe
ln -s ../../caffe
# pre-trained caffe model (trained model with stage 1 training)
ln -s ../../model
# training result saving path
ln -s ../../model/stage_2_train_result

## create directories
mkdir snapshot
mkdir training_log

## start training
./start_train.sh

## copy and rename trained model
cp ./snapshot/stage_2_train_iter_40000.caffemodel ./stage_2_train_result/stage_2_train_result.caffemodel

