% -----------------------------------------------------------------------------------------------------------------------
function [imout_z, imout_x, labels, sizes_z, sizes_x] = vid_get_random_batch(imdb, imdb_video, batch, data_dir, varargin)
%VID_GET_RANDOM_BATCH
%   returns batch of pairs of input (z and x) and labels
%   (Sizes are returned as [height, width])
%
%   Luca Bertinetto, Jack Valmadre, Joao Henriques, 2016
% -----------------------------------------------------------------------------------------------------------------------
    % Defines
    RGB = 1;
    GRAY = 2;
    TRAIN_SET = 1;
    VAL_SET = 2;
    % Default parameters
    opts.exemplarSize = [];
    opts.instanceSize = [];
    opts.frameRange = 50;
    opts.subMean = false;
    opts.colorRange = 255; % Adjust range from [0, 255] to [0, colorRange].
    opts.stats.rgbMean_z = [];
    opts.stats.rgbVariance_z = [];
    opts.stats.rgbMean_x = [];
    opts.stats.rgbVariance_x = [];
    opts.augment.translate = false;
    opts.augment.maxTranslate = []; % empty means no max
    opts.augment.stretch = false;
    opts.augment.maxStretch = 0.1;
    opts.augment.color = false;
    opts.augment.grayscale = 0;
    opts.prefetch = false;
    opts.numThreads = 12;
    opts = vl_argparse(opts, varargin);
% -----------------------------------------------------------------------------------------------------------------------
    % Determine the set (e.g. train or val) of the batch.
    batch_set = imdb.images.set(batch(1));
    % Check all images in the batch are from the same set.
    assert(all(batch_set == imdb.images.set(batch)));
    batch_size = numel(batch);
    % Decide rgb vs gray with given probabilities
    pair_types_rgb = datasample(1:2, batch_size, 'Weights', [1-opts.augment.grayscale opts.augment.grayscale]);
% -----------------------------------------------------------------------------------------------------------------------
    % randomize a subset of the imdb of batch_size - decide set of videos to sample from
    % NOTE: the purpose of imdb is to preserve compatibility with MatConvNet cnn_train_dag,
    % it just defines size of an epoch and the type of batch (train/val).
    ids_set = find(imdb_video.set==batch_set);
    % sample one video for each pair
    rnd_videos = datasample(ids_set, batch_size, 'Replace', false);
    ids_pairs = rnd_videos(1:batch_size);
    % Initialize pairs
    % First load location for all pairs, all images will be loaded at once with vl_imreadjpeg
    % objects contains metadata for all the pairs of the batch
    objects = struct();
    objects.set = batch_set * uint8(ones(1, batch_size));
    objects.z = cell(1, batch_size);
    objects.x = cell(1, batch_size);
    % crops locations
    crops_z_string = cell(1, batch_size);
    crops_x_string = cell(1, batch_size);
    labels = zeros(1, batch_size);
    % final augmented crops
    imout_z = zeros(opts.exemplarSize, opts.exemplarSize, 3, batch_size, 'single');
    imout_x = zeros(opts.instanceSize, opts.instanceSize, 3, batch_size, 'single');

    for i = 1:batch_size
        % Crops from same videos, centered on the object
        labels(i) = 1;
        [objects.z{i}, objects.x{i}] = choose_pos_pair(imdb_video, ids_pairs(i), opts.frameRange);
    end

    % get absolute paths of crops locations
    for i=1:batch_size
        crops_z_string{i} = [strrep(fullfile(data_dir, objects.z{i}.frame_path), '.JPEG','') '.' num2str(objects.z{i}.track_id, '%02d') '.crop.z.jpg'];
        crops_x_string{i} = [strrep(fullfile(data_dir, objects.x{i}.frame_path), '.JPEG','') '.' num2str(objects.x{i}.track_id, '%02d') '.crop.x.jpg'];
    end
    % prepare all the files to read
    files = [crops_z_string crops_x_string];

    % prefetch is used to load images in a separate thread
    if opts.prefetch
        error('to implement');
    end

    % read all the crops efficiently
    crops = vl_imreadjpeg(files, 'numThreads', opts.numThreads);
    crops_z = crops(1:batch_size);
    crops_x = crops(batch_size+1 : end);
    clear crops
    % -----------------------------------------------------------------------------------------------------------------------
    % Data augmentation
    % Only augment during training.
    if batch_set == TRAIN_SET
        aug_opts = opts.augment;
    else
        aug_opts = struct('translate', false, ...
                          'maxTranslate', 0, ...
                          'stretch', false, ...
                          'maxStretch', 0, ...
                          'color', false);
    end

    aug_z = @(crop) acquire_augment(crop, opts.exemplarSize, opts.stats.rgbVariance_z, aug_opts);
    aug_x = @(crop) acquire_augment(crop, opts.instanceSize, opts.stats.rgbVariance_x, aug_opts);

    for i=1:batch_size

        tmp_z = aug_z(crops_z{i});
        tmp_x = aug_x(crops_x{i});

        switch pair_types_rgb(i)
            case RGB
                imout_z(:,:,:,i) = tmp_z;
                imout_x(:,:,:,i) = tmp_x;
            case GRAY
                % vl_imreadjpeg returns images in [0, 255] with class single.
                imout_z(:,:,:,i) = repmat(rgb2gray(tmp_z/255)*255, [1 1 3]);
                imout_x(:,:,:,i) = repmat(rgb2gray(tmp_x/255)*255, [1 1 3]);
        end

        if opts.subMean
            % Sanity check - mean should be in range 0-255!
            means = [opts.stats.rgbMean_z(:); opts.stats.rgbMean_x(:)];
            lower = 0.2 * 255;
            upper = 0.8 * 255;
            if ~all((lower <= means) & (means <= upper))
                error('mean does not seem to for pixels in 0-255');
            end
            imout_z = bsxfun(@minus, imout_z, reshape(opts.stats.rgbMean_z, [1 1 3]));
            imout_x = bsxfun(@minus, imout_x, reshape(opts.stats.rgbMean_x, [1 1 3]));
        end
        imout_z = imout_z / 255 * opts.colorRange;
        imout_x = imout_x / 255 * opts.colorRange;
    end

    sizes_z = zeros(2, batch_size);
    sizes_x = zeros(2, batch_size);
    for i = 1:batch_size
        if ~(labels(i) > 0)
            continue
        end

        % compute bounding boxes of objects within crops x and z
        [bbox_z, bbox_x] = get_objects_extent(double(objects.z{i}.extent), double(objects.x{i}.extent), opts.exemplarSize, opts.instanceSize);
        % only store h and w
        sizes_z(:,i) = bbox_z([4 3]);
        sizes_x(:,i) = bbox_x([4 3]);
    end
end

% -----------------------------------------------------------------------------------------------------------------------
function [z, x] = choose_pos_pair(imdb_video, rand_vid, frameRange)
% Get positive pair with crops from same videos, centered on the object
% -----------------------------------------------------------------------------------------------------------------------
    valid_trackids = find(imdb_video.valid_trackids(:, rand_vid) > 1);
    assert(~isempty(valid_trackids), 'No valid trackids for a video in the batch.');
    rand_trackid_z = datasample(valid_trackids, 1);
    % pick valid exemplar from the random trackid
    rand_z = datasample(imdb_video.valid_per_trackid{rand_trackid_z, rand_vid}, 1);
    % pick valid instance within frameRange seconds from the exemplar, excluding the exemplar itself
    possible_x_pos = (1:numel(imdb_video.valid_per_trackid{rand_trackid_z, rand_vid}));
    [~, rand_z_pos] = ismember(rand_z, imdb_video.valid_per_trackid{rand_trackid_z, rand_vid});
    possible_x_pos = possible_x_pos([max(rand_z_pos-frameRange, 1):(rand_z_pos-1), (rand_z_pos+1):min(rand_z_pos+frameRange, numel(possible_x_pos))]);
    possible_x = imdb_video.valid_per_trackid{rand_trackid_z, rand_vid}(possible_x_pos);
    assert(~isempty(possible_x), 'No valid x for the chosen z.');
    rand_x = datasample(possible_x, 1);
    assert(imdb_video.objects{rand_vid}{rand_x}.valid, 'error picking rand x.');
    z = imdb_video.objects{rand_vid}{rand_z};
    x = imdb_video.objects{rand_vid}{rand_x};
end

% -----------------------------------------------------------------------------------------------------------------------
function imo = acquire_augment(im, imageSize, rgbVariance, aug_opts)
% Apply transformations and augmentations to original crops
% -----------------------------------------------------------------------------------------------------------------------
    if numel(imageSize) == 1
        imageSize = [imageSize, imageSize];
    end
    if numel(aug_opts.maxTranslate) == 1
        aug_opts.maxTranslate = [aug_opts.maxTranslate, aug_opts.maxTranslate];
    end

    imt = im;
    if size(imt,3) == 1
        imt = cat(3, imt, imt, imt);
    end

    w = size(imt,2) ;
    h = size(imt,1) ;
    cx = (w+1)/2;
    cy = (h+1)/2;

    if aug_opts.stretch
        scale = (1+aug_opts.maxStretch*(-1+2*rand(2,1)));
        sz = round(min(imageSize(1:2)' .* scale, [h;w]));
    else
        sz = imageSize;
    end

    if aug_opts.translate
        if isempty(aug_opts.maxTranslate)
            % Take any crop within the image.
            dx = randi(w - sz(2) + 1, 1) ;
            dy = randi(h - sz(1) + 1, 1) ;
        else
            % Take crop within maxTranslate of center.
            mx = min(aug_opts.maxTranslate(2), floor((w-sz(2))/2));
            my = min(aug_opts.maxTranslate(1), floor((h-sz(1))/2));
            % Check bounds:
            % dx = (w+1)/2 - (sz(2)-1)/2 - (w-sz(2))/2
            %    = (w+1 - sz(2)+1 - w+sz(2))/2
            %    = 1 + (w - sz(2) - w+sz(2))/2
            %    = 1
            % dx + sz(2)-1 = (w+1)/2 - (sz(2)-1)/2 + (w-sz(2))/2 + sz(2)-1
            %              = (w+1 - sz(2)+1 + w-sz(2) + 2*sz(2)-2)/2
            %              = (w - sz(2) + w-sz(2) + 2*sz(2))/2
            %              = (w + w)/2
            %              = w
            dx = cx - (sz(2)-1)/2 + randi([-mx, mx], 1);
            dy = cy - (sz(1)-1)/2 + randi([-my, my], 1);
        end
    else
        % Take crop at center.
        dx = cx - (sz(2)-1)/2;
        dy = cy - (sz(1)-1)/2;
    end
    sx = round(linspace(dx, dx+sz(2)-1, imageSize(2))) ;
    sy = round(linspace(dy, dy+sz(1)-1, imageSize(1))) ;

    % flip = rand > 0.5 ;
    % if flip, sx = fliplr(sx) ; end

    if ~aug_opts.color
        imo = imt(sy,sx,:);
    else
        offset = reshape(rgbVariance * randn(3,1), 1,1,3);
        imo = bsxfun(@minus, imt(sy,sx,:), offset);
    end
end

% -----------------------------------------------------------------------------------------------------------------------
function [bbox_z, bbox_x] = get_objects_extent(object_z_extent, object_x_extent, size_z, size_x)
% Compute objects bbox within crops
% bboxes are returned as [xmin, ymin, width, height]
% -----------------------------------------------------------------------------------------------------------------------
    % TODO: this should passed from experiment as default
    context_amount = 0.5;
    
    % getting in-crop object extent for Z
    [w_z, h_z] = deal(object_z_extent(3), object_z_extent(4));
    wc_z = w_z + context_amount*(w_z+h_z);   
    hc_z = h_z + context_amount*(w_z+h_z);
    s_z = sqrt(wc_z*hc_z);
    scale_z = size_z / s_z;   
    ws_z = w_z * scale_z;
    hs_z = h_z * scale_z;
    bbox_z = [(size_z-ws_z)/2, (size_z-hs_z)/2, ws_z, hs_z];
    
    % getting in-crop object extent for X    
    [w_x, h_x] = deal(object_x_extent(3), object_x_extent(4));
    wc_x = w_x + context_amount*(w_x+h_x);   
    hc_x = h_x + context_amount*(w_x+h_x);
    s_xz = sqrt(wc_x*hc_x);
    scale_xz = size_z / s_xz;
    
    d_search = (size_x - size_z)/2;
    pad = d_search/scale_xz;
    s_x = s_xz + 2*pad;
    scale_x = size_x / s_x;
    ws_x = w_x * scale_x;
    hs_x = h_x * scale_x;
    bbox_x = [(size_x-ws_x)/2, (size_x-hs_x)/2, ws_x, hs_x];
end
