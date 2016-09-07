% ------------------------------------------------------------------------
function stats = vid_image_stats(imdb_video, perc_training, base_path)
%VID_IMAGE_STATS
%	Compute basic colour stats for a random PERC_TRAINING of the dataset
%	Used for data augmentation during training.
% 	e.g. vid_image_stats(imdb_video, 0.1, /path/to/curated/ILSVRC15/')
% ------------------------------------------------------------------------
    % collect different stats for z and x crops (x contains more padding)
    z_sz = 127;
    x_sz = 255;
    samples_per_video = 16;
    imout_z = zeros(z_sz, z_sz, 3, samples_per_video, 'single');
    imout_x = zeros(x_sz, x_sz, 3, samples_per_video, 'single');
    crops_z_string = cell(1, samples_per_video);
    crops_x_string = cell(1, samples_per_video);
    n_video = numel(imdb_video.id);
    n_video_train = round(perc_training * n_video);
    avg_z = cell(1,n_video_train);
    rgbm1_z = cell(1,n_video_train);
    rgbm2_z = cell(1,n_video_train);
    n_z = samples_per_video*z_sz*z_sz;
    avg_x = cell(1,n_video_train);
    rgbm1_x = cell(1,n_video_train);
    rgbm2_x = cell(1,n_video_train);
    n_x = samples_per_video*x_sz*x_sz;
    for v=1:n_video_train
        n_obj = numel(imdb_video.objects{v});
        rand_objs = datasample(1:n_obj, samples_per_video);
        for o=1:samples_per_video
            crops_z_string{o} = [base_path strrep(imdb_video.objects{v}{rand_objs(o)}.frame_path, '.JPEG','') '.' num2str(imdb_video.objects{v}{rand_objs(o)}.track_id, '%02d') '.crop.z.jpg'];
            crops_x_string{o} = [base_path strrep(imdb_video.objects{v}{rand_objs(o)}.frame_path, '.JPEG','') '.' num2str(imdb_video.objects{v}{rand_objs(o)}.track_id, '%02d') '.crop.x.jpg'];
        end
        files = [crops_z_string crops_x_string];
        imgs = vl_imreadjpeg(files, 'numThreads', 12);
        crops_z = imgs(1:samples_per_video);
        crops_x = imgs(samples_per_video+1 : end);
        for o=1:samples_per_video
           imout_z(:,:,:,o) = crops_z{o};
           imout_x(:,:,:,o) = crops_x{o};
        end

        Z = reshape(permute(imout_z,[3 1 2 4]),3,[]);
        avg_z{end+1} = mean(imout_z, 4);
        rgbm1_z{end+1} = sum(Z,2)/n_z;
        rgbm2_z{end+1} = Z*Z'/n_z;

        X = reshape(permute(imout_x,[3 1 2 4]),3,[]);
        avg_x{end+1} = mean(imout_x, 4);
        rgbm1_x{end+1} = sum(X,2)/n_x;
        rgbm2_x{end+1} = X*X'/n_x;
        fprintf('Processed video %d/%d\n', v, n_video_train);
    end

    stats = struct();
    stats.z = struct();
    stats.x = struct();

    stats.z.averageImage = mean(cat(4,avg_z{:}),4);
    rgbm1_z = mean(cat(2,rgbm1_z{:}),2);
    stats.z.rgbm1 = rgbm1_z;
    rgbm2_z = mean(cat(3,rgbm2_z{:}),3);
    stats.z.rgbm2 = rgbm2_z;
    stats.z.rgbMean = rgbm1_z;
    stats.z.rgbCovariance = rgbm2_z - rgbm1_z*rgbm1_z';

    stats.x.averageImage = mean(cat(4,avg_x{:}),4);
    rgbm1_x = mean(cat(2,rgbm1_x{:}),2);
    stats.x.rgbm1 = rgbm1_x;
    rgbm2_x = mean(cat(3,rgbm2_x{:}),3);
    stats.z.rgbm2 = rgbm2_z;
    stats.x.rgbMean = rgbm1_x;
    stats.x.rgbCovariance = rgbm2_x - rgbm1_x*rgbm1_x';
end
