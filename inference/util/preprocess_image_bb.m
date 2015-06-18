function preprocessed_img = preprocess_image_bb(img, box, img_sz)
meanImg = [104.00698793, 116.66876762, 122.67891434]; % order = bgr
meanImg = repmat(meanImg, [img_sz^2,1]);
meanImg = reshape(meanImg, [img_sz, img_sz, 3]); 

crop = double(img(box(2):box(4),box(1):box(3),:));
crop = imresize(crop, [img_sz img_sz], 'bilinear'); % resize cropped image
crop = crop(:,:,[3 2 1]) - meanImg; % convert color channer rgb->bgr and subtract mean 
preprocessed_img = {single(permute(crop, [2 1 3]))}; 

end