LOGDIR=./training_log
CAFFE=./caffe/build/tools/caffe
SOLVER=./solver.prototxt
WEIGHTS=./model/VGG_conv/VGG_ILSVRC_16_layers_conv.caffemodel

GLOG_log_dir=$LOGDIR $CAFFE train -solver $SOLVER -weights $WEIGHTS -gpu 0

#send_notify_mail "039_voc_single_object_finetune_from_031 train script is finished"
