# get caffe (you should compile this caffe library with your own Makefile.config file)
git clone https://github.com/HyeonwooNoh/caffe.git

# get data
cd data
./get_data.sh
cd ..

# get models
cd model
./get_model.sh
cd ..

# prepare inference
cd inference
./setup.sh
cd ..

