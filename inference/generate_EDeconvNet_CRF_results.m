function generate_EDeconvNet_CRF_results(config)

fprintf('start generating EDeconvNet+CRF results\n');

%% initialization
load(config.cmap);
init_VOC2012_TEST;

%% initialize caffe
addpath(fullfile(config.Path.CNN.caffe_root, 'matlab/caffe'));
fprintf('initializing caffe..\n');
if caffe('is_initialized')
    caffe('reset')
end
caffe('init', config.Path.CNN.model_proto, config.Path.CNN.model_data)
caffe('set_device', config.gpuNum);
caffe('set_mode_gpu');
caffe('set_phase_test');
fprintf('done\n');


%% initialize paths
save_res_dir = [config.save_root, '/EDeconvNet_CRF'];
save_res_path = [save_res_dir, '/%s.png'];
edgebox_cache_path = [config.edgebox_cache_dir '/%s.mat'];

fcn_score_dir = [config.fcn_score_dir '/scores'];
fcn_score_path = [fcn_score_dir '/%s.mat'];

%% create directory
if config.write_file
    if ~exist(save_res_dir), mkdir(save_res_dir), end
end


fprintf('start generating result\n');
fprintf('caffe model: %s\n', config.Path.CNN.model_proto);
fprintf('caffe weight: %s\n', config.Path.CNN.model_data);


%% read VOC2012 TEST image set
ids=textread(sprintf(VOCopts.seg.imgsetpath, config.imageset), '%s');

for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...', i, length(ids), ids{i});  
    tic;
    
    % read image
    I=imread(sprintf(VOCopts.imgpath,ids{i}));
    
    result_base = uint8(zeros(size(I,1), size(I,2)));
    prob_base = zeros(size(I,1),size(I,2),21);
    cnt_base = uint8(zeros(size(I,1),size(I,2)));
    norm_prob_base = zeros(size(I,1),size(I,2),21);
        
    % padding for easy cropping    
    [img_height, img_width, ~] = size(I);
    pad_offset_col = img_height;
    pad_offset_row = img_width;
    
    % pad every images(I, cls_seg, inst_seg...) to make cropping easy
    padded_I = padarray(I,[pad_offset_row, pad_offset_col]);
    padded_result_base = padarray(result_base,[pad_offset_row, pad_offset_col]);
    padded_prob_base = padarray(prob_base,[pad_offset_row, pad_offset_col]);
    padded_cnt_base = padarray(cnt_base, [pad_offset_row, pad_offset_col]);
    norm_padded_prob_base = padarray(norm_prob_base,[pad_offset_row, pad_offset_col]);
    norm_padded_prob_base(:,:,1) = eps;
        
    padded_frame_255 = 255-padarray(uint8(ones(size(I,1),size(I,2))*255),[pad_offset_row, pad_offset_col]);
    
    padded_result_base = padded_result_base + padded_frame_255;

    %% load extended bounding box
    cache = load(sprintf(edgebox_cache_path, ids{i})); % boxes_padded
    boxes_padded = cache.boxes_padded;
    
    numBoxes = size(boxes_padded,1);    
    cnt_process = 1;
    for bidx = 1:numBoxes  
        box = boxes_padded(bidx,:);
        box_wd = box(3)-box(1)+1;
        box_ht = box(4)-box(2)+1;
        
        if min(box_wd, box_ht) < 112, continue; end   
        
        input_data = preprocess_image_bb(padded_I, boxes_padded(bidx,:), config.im_sz); 
        cnn_output = caffe('forward', input_data);
        
        segImg = permute(cnn_output{1}, [2, 1, 3]);
        segImg = imresize(segImg, [box_ht, box_wd], 'bilinear');
        
        % accumulate prediction result
        cropped_prob_base = padded_prob_base(box(2):box(4),box(1):box(3),:);
        padded_prob_base(box(2):box(4),box(1):box(3),:) = max(cropped_prob_base,segImg);
        
        if mod(cnt_process, 10) == 0, fprintf(',%d', cnt_process); end
        if cnt_process >= config.max_proposal_num
            break;
        end
        
        cnt_process = cnt_process + 1;
    end
    
    %% save DeconvNet prediction score
    deconv_score = padded_prob_base(pad_offset_row:pad_offset_row+size(I,1)-1,pad_offset_col:pad_offset_col+size(I,2)-1,:);
    
    %% load fcn-8s score
    fcn_cache = load(sprintf(fcn_score_path, ids{i}));
    fcn_score = fcn_cache.score;
    
    %% ensemble
    zero_mask = zeros(size(fcn_score));
    fcn_score = max(zero_mask,fcn_score);
        
    ens_score = deconv_score .* fcn_score;
    [ens_segscore, ens_segmask] = max(ens_score, [], 3);
    ens_segmask = uint8(ens_segmask-1);
    
    %% densecrf
    fprintf('[densecrf.. ');
    prob_map = exp(ens_score - repmat(max(ens_score, [], 3), [1,1,size(ens_score,3)]));
    prob_map = prob_map ./ repmat(sum(prob_map, 3), [1,1, size(prob_map,3)]);
    unary = -log(prob_map);

    D = Densecrf(I,single(unary));

    % Some settings.
    D.gaussian_x_stddev = 3;
    D.gaussian_y_stddev = 3;
    D.gaussian_weight = 3; 

    D.bilateral_x_stddev = 20;
    D.bilateral_y_stddev = 20;
    D.bilateral_r_stddev = 3;
    D.bilateral_g_stddev = 3;
    D.bilateral_b_stddev = 3;
    D.bilateral_weight = 5;     
   
    D.iterations = 10;

    D.mean_field;
    segmask = D.segmentation;
    resulting_seg = uint8(segmask-1);    
    fprintf('done] ');
    
    %% save or display result
    if config.write_file
        imwrite(resulting_seg,cmap,sprintf(save_res_path, ids{i}));
    else
        subplot(1,2,1);
        imshow(I);
        subplot(1,2,2);
        resulting_seg_im = reshape(cmap(int32(resulting_seg)+1,:),[size(resulting_seg,1),size(resulting_seg,2),3]);
        imshow(resulting_seg_im);
        waitforbuttonpress;
    end
    fprintf(' done [%f]\n', toc);
end

%% function end
end
