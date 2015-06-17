##########################
# start stage 1 training #
##########################
cd 001_stage_1

## create simulinks

# image data
ln -s ../../data/VOC2012  
# training / validataion imageset
ln -s ../../data/imagesets/stage_1_train_imgset
# caffe
ln -s ../../caffe
# pre-trained caffe model (VGG16 - fully convolution)
ln -s ../../model
# training result saving path
ln -s ../../model/stage_1_train_result

## create directories
mkdir snapshot
mkdir training_log

## start training
./start_train.sh

## copy and rename trained model
cp ./snapshot/stage_1_train_iter_20000.caffemodel ./stage_1_train_result/stage_1_train_result.caffemodel
