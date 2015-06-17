LOGDIR=./testing_log
CAFFE=./caffe/build/tools/caffe
MODEL=./DeconvNet_inference_test.prototxt
WEIGHTS=./DeconvNet_trainval_inference.caffemodel

GLOG_log_dir=$LOGDIR $CAFFE test -model $MODEL -weights $WEIGHTS -gpu 0


