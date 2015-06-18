function preprocessed_img = preprocess_image(img, img_sz)
% B 104.00698793 G 116.66876762 R 122.67891434
mean = [104.00698793; 116.66876762; 122.67891434];
mean = permute(mean,[2,3,1]);

I_zero_mean = double(img(:,:,[3 2 1])) - repmat(mean,[size(img,1),size(img,2),1]);
im = padarray(I_zero_mean,[img_sz - size(I_zero_mean,1), img_sz - size(I_zero_mean,2)],'post');

caffe_images = zeros(img_sz, img_sz, 3, 1, 'single');
caffe_images(:,:,:,1) = single(permute(im,[2, 1, 3]));

preprocessed_img = {caffe_images};
end