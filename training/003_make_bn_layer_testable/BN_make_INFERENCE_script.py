
### User Configuration

iteration = 500
gpu_id_num = 0

## path configuration
caffe_root = './caffe'
script_path = '.'
caffe_model = script_path + '/DeconvNet_make_bn_layer_testable.prototxt'
caffe_weight = script_path + '/model/stage_2_train_result/stage_2_train_result.caffemodel'
caffe_inference_weight = script_path + '/DeconvNet_trainval_inference.caffemodel'

## modify this definition according to model definition
bn_blobs= ['conv1_1','conv1_2',
           'conv2_1','conv2_2',
           'conv3_1','conv3_2','conv3_3',
           'conv4_1','conv4_2','conv4_3', 
           'conv5_1','conv5_2','conv5_3', 
           'fc6','fc7',
           'fc6-deconv',
           'deconv5_1','deconv5_2','deconv5_3',
           'deconv4_1','deconv4_2','deconv4_3', 
           'deconv3_1','deconv3_2','deconv3_3',
           'deconv2_1','deconv2_2',
           'deconv1_1','deconv1_2']
bn_layers=['bn1_1','bn1_2',
           'bn2_1','bn2_2',
           'bn3_1','bn3_2','bn3_3',
           'bn4_1','bn4_2','bn4_3',
           'bn5_1','bn5_2','bn5_3',
           'bnfc6','bnfc7',
           'fc6-deconv-bn',
           'debn5_1','debn5_2','debn5_3',
           'debn4_1','debn4_2','debn4_3',
           'debn3_1','debn3_2','debn3_3',
           'debn2_1','debn2_2',
           'debn1_1','debn1_2']
bn_means= ['bn1_1-mean','bn1_2-mean',
           'bn2_1-mean','bn2_2-mean',
           'bn3_1-mean','bn3_2-mean','bn3_3-mean',
           'bn4_1-mean','bn4_2-mean','bn4_3-mean',
           'bn5_1-mean','bn5_2-mean','bn5_3-mean',
           'bnfc6-mean','bnfc7-mean',
           'fc6-deconv-bn-mean',
           'debn5_1-mean','debn5_2-mean','debn5_3-mean',
           'debn4_1-mean','debn4_2-mean','debn4_3-mean',
           'debn3_1-mean','debn3_2-mean','debn3_3-mean',
           'debn2_1-mean','debn2_2-mean',
           'debn1_1-mean','debn1_2-mean']
bn_vars = ['bn1_1-var','bn1_2-var',
           'bn2_1-var','bn2_2-var',
           'bn3_1-var','bn3_2-var','bn3_3-var',
           'bn4_1-var','bn4_2-var','bn4_3-var',
           'bn5_1-var','bn5_2-var','bn5_3-var',
           'bnfc6-var','bnfc7-var',
           'fc6-deconv-bn-var',
           'debn5_1-var','debn5_2-var','debn5_3-var',
           'debn4_1-var','debn4_2-var','debn4_3-var',
           'debn3_1-var','debn3_2-var','debn3_3-var',
           'debn2_1-var','debn2_2-var',
           'debn1_1-var','debn1_2-var']


### start generate caffemodel

print 'start generating BN-testable caffemodel'
print 'caffe_root: %s' % caffe_root
print 'script_path: %s' % script_path
print 'caffe_model: %s' % caffe_model
print 'caffe_weight: %s' % caffe_weight
print 'caffe_inference_weight: %s' % caffe_inference_weight

import numpy as np

import sys
sys.path.append(caffe_root+'/python')
import caffe
from caffe.proto import caffe_pb2

net = caffe.Net(caffe_model, caffe_weight)
#net.set_mode_cpu()
net.set_mode_gpu()
net.set_device(gpu_id_num)
net.set_phase_test()

def forward_once(net):
    start_ind = 0
    end_ind = len(net.layers) - 1
    net._forward(start_ind, end_ind)
    return {out: net.blobs[out].data for out in net.outputs}

print net.params.keys()

res = forward_once(net)

bn_avg_mean = {bn_mean: np.squeeze(res[bn_mean]).copy() for bn_mean in bn_means}
bn_avg_var = {bn_var: np.squeeze(res[bn_var]).copy() for bn_var in bn_vars}    

cnt = 1

for i in range(0, iteration):
    res = forward_once(net)
    for bn_mean in bn_means:
        bn_avg_mean[bn_mean] = bn_avg_mean[bn_mean] + np.squeeze(res[bn_mean])
    for bn_var in bn_vars:
        bn_avg_var[bn_var] = bn_avg_var[bn_var] + np.squeeze(res[bn_var])
        
    cnt += 1
    print 'progress: %d/%d' % (i, iteration)

## compute average
for bn_mean in bn_means:
    bn_avg_mean[bn_mean] /= cnt
for bn_var in bn_vars:
    bn_avg_var[bn_var] /= cnt

for i in range(0, len(bn_vars)):
    m = np.prod(net.blobs[bn_blobs[i]].data.shape) / np.prod(bn_avg_var[bn_vars[i]].shape)
    bn_avg_var[bn_vars[i]] *= (m/(m-1))

scale_data = {bn_layer: np.squeeze(net.params[bn_layer][0].data) for bn_layer in bn_layers}
shift_data = {bn_layer: np.squeeze(net.params[bn_layer][1].data) for bn_layer in bn_layers}

var_eps = 1e-9

new_scale_data = {}
new_shift_data = {}
for i in range(0, len(bn_layers)):
    gamma = scale_data[bn_layers[i]]
    beta = shift_data[bn_layers[i]]
    Ex = bn_avg_mean[bn_means[i]]
    Varx = bn_avg_var[bn_vars[i]]
    new_gamma = gamma / np.sqrt(Varx + var_eps)
    new_beta = beta - (gamma * Ex / np.sqrt(Varx + var_eps))
    
    new_scale_data[bn_layers[i]] = new_gamma
    new_shift_data[bn_layers[i]] = new_beta

print new_scale_data.keys()
print new_shift_data.keys()

## assign computed new scale and shift values to net.params
for i in range(0, len(bn_layers)):
    net.params[bn_layers[i]][0].data[...] = new_scale_data[bn_layers[i]].reshape(net.params[bn_layers[i]][0].data.shape)
    net.params[bn_layers[i]][1].data[...] = new_shift_data[bn_layers[i]].reshape(net.params[bn_layers[i]][1].data.shape)

print 'start saving model'

net.save(caffe_inference_weight)

print 'done'



