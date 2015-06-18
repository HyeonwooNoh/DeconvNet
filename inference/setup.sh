############################
# prepare to run inference #
############################

# create simulinks
# caffe
ln -s ../caffe
ln -s ../model/DeconvNet
ln -s ../model/FCN

# download necessary data for inference
cd data
# VOC2012 test data
wget http://cvlab.postech.ac.kr/research/deconvnet/data/VOC2012_TEST.tar.gz
tar -zxvf VOC2012_TEST.tar.gz
rm -rf VOC2012_TEST.tar.gz

# edgebox cached data for VOC2012 test
wget http://cvlab.postech.ac.kr/research/deconvnet/data/edgebox_cached.tar.gz
tar -zxvf edgebox_cached.tar.gz
rm -rf edgebox_cached.tar.gz

cd ..

# download external dependencies
cd ext
# densecrf
git clone https://github.com/johannesu/meanfield-matlab.git densecrf
cd ..
