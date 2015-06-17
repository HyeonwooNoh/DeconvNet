################################
# start make bn layer testable #
################################
cd 003_make_bn_layer_testable

## create simulinks

# image data
ln -s ../../data/VOC2012
# training / validataion imageset
ln -s ../../data/imagesets/stage_2_train_imgset
# caffe
ln -s ../../caffe
# pre-trained caffe model (trained model with stage 2 training)
ln -s ../../model
ln -s ../../model/DeconvNet

## create directories
mkdir testing_log

## start make bn layer testable
python BN_make_INFERENCE_script.py

## run test to check
./start_test.sh

## copy and rename trained model
cp ./DeconvNet_trainval_inference.caffemodel ./DeconvNet/DeconvNet_trainval_inference.caffemodel

