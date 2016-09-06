% -------------------------------------------------------------------------------------------------------------------
function save_crops(imdb_video,v_1,v_end, root_original, root_crops)
    % Extract and save crops from video v_1 (start from 1) to v_end (check num video in imdb)
	% e.g. save_crops(imdb_video, 1, 1000, '/path/to/original/ILSVRC15/', '/path/to/new/curated/ILSVRC15/')
% -------------------------------------------------------------------------------------------------------------------
    rootDataDir_src = [root_original '/Data/VID/train/'];
    rootDataDir_dest = [root_crops '/Data/VID/train/'];
    % sides of the crops for z and x saved on disk
    exemplar_size = 127;
    instance_size = 255;
    context_amount = 0.5;

    saved_crops = 0;
    for v=v_1:v_end
        valid_trackids = find(imdb_video.valid_trackids(:,v));
        for ti=1:numel(valid_trackids)
            valid_objects = imdb_video.valid_per_trackid{valid_trackids(ti),v};
            for o = 1:numel(valid_objects)
                obj = imdb_video.objects{v}{valid_objects(o)};
                assert(obj.valid)
                fprintf('%d %d: %s\n', v, obj.track_id, obj.frame_path);
                im = imread([rootDataDir_src obj.frame_path]);

                [im_crop_z, bbox_z, pad_z, im_crop_x, bbox_x, pad_x] = get_crops(im, obj, exemplar_size, instance_size, context_amount);

                root = [rootDataDir_dest strrep(obj.frame_path,'.JPEG','') '.' num2str(obj.track_id,'%02d')];
                pz = fopen([root '.pad.z.txt'],'w');
                px = fopen([root '.pad.x.txt'],'w');
                fprintf(pz,'%.2f,%.2f,%.2f,%.2f\n', pad_z(1),pad_z(2),pad_z(3),pad_z(4));
                fprintf(px,'%.2f,%.2f,%.2f,%.2f\n', pad_x(1),pad_x(2),pad_x(3),pad_x(4));
                fclose(pz);
                fclose(px);
                imwrite(im_crop_z, [root '.crop.z.jpg'], 'Quality', 90);
                imwrite(im_crop_x, [root '.crop.x.jpg'], 'Quality', 90);

                saved_crops = saved_crops+1;
            end
        end
    end

    fprintf('\n:: SAVED %d crops ::\n', saved_crops);

end

% -------------------------------------------------------------------------------------------------------------------
function [im_crop_z, bbox_z, pad_z, im_crop_x, bbox_x, pad_x] = get_crops(im, object, size_z, size_x, context_amount)
% -------------------------------------------------------------------------------------------------------------------
    %% Get exemplar sample
    % take bbox with context for the exemplar

    bbox = double(object.extent);
    [cx, cy, w, h] = deal(bbox(1)+bbox(3)/2, bbox(2)+bbox(4)/2, bbox(3), bbox(4));
    wc_z = w + context_amount*(w+h);
    hc_z = h + context_amount*(w+h);
    s_z = sqrt(single(wc_z*hc_z));
    scale_z = size_z / s_z;
    [im_crop_z, left_pad_z, top_pad_z, right_pad_z, bottom_pad_z] = get_subwindow_avg(im, [cy cx], [size_z size_z], [round(s_z) round(s_z)]);
    pad_z = ceil([scale_z*(left_pad_z+1) scale_z*(top_pad_z+1) size_z-scale_z*(right_pad_z+left_pad_z) size_z-scale_z*(top_pad_z+bottom_pad_z+1)]);
    %% Get instance sample
    d_search = (size_x - size_z)/2;
    pad = d_search/scale_z;
    s_x = s_z + 2*pad;
    scale_x = size_x / s_x;
    [im_crop_x, left_pad_x, top_pad_x, right_pad_x, bottom_pad_x] = get_subwindow_avg(im, [cy cx], [size_x size_x], [round(s_x) round(s_x)]);
    pad_x = ceil([scale_x*(left_pad_x+1) scale_x*(top_pad_x+1) size_x-scale_x*(right_pad_x+left_pad_x) size_x-scale_x*(top_pad_x+bottom_pad_x+1)]);
    % Size of object within the crops
    ws_z = w * scale_z;
    hs_z = h * scale_z;
    ws_x = w * scale_x;
    hs_x = h * scale_x;
    bbox_z = [(size_z-ws_z)/2, (size_z-hs_z)/2, ws_z, hs_z];
    bbox_x = [(size_x-ws_x)/2, (size_x-hs_x)/2, ws_x, hs_x];
end

% ---------------------------------------------------------------------------------------------------------------
function [im_patch, left_pad, top_pad, right_pad, bottom_pad] = get_subwindow_avg(im, pos, model_sz, original_sz)
%GET_SUBWINDOW_AVG Obtain image sub-window, padding with avg channel if area goes outside of border
% ---------------------------------------------------------------------------------------------------------------

    avg_chans = [mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))];

    if isempty(original_sz)
        original_sz = model_sz;
    end
    sz = original_sz;
    im_sz = size(im);
    %make sure the size is not too small
    assert(all(im_sz(1:2) > 2));
    c = (sz+1) / 2;

    %check out-of-bounds coordinates, and set them to avg_chans
    context_xmin = round(pos(2) - c(2));
    context_xmax = context_xmin + sz(2) - 1;
    context_ymin = round(pos(1) - c(1));
    context_ymax = context_ymin + sz(1) - 1;
    left_pad = double(max(0, 1-context_xmin));
    top_pad = double(max(0, 1-context_ymin));
    right_pad = double(max(0, context_xmax - im_sz(2)));
    bottom_pad = double(max(0, context_ymax - im_sz(1)));

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;

    if top_pad || left_pad
        R = padarray(im(:,:,1), [top_pad left_pad], avg_chans(1), 'pre');
        G = padarray(im(:,:,2), [top_pad left_pad], avg_chans(2), 'pre');
        B = padarray(im(:,:,3), [top_pad left_pad], avg_chans(3), 'pre');
        im = cat(3, R, G, B);
    end

    if bottom_pad || right_pad
        R = padarray(im(:,:,1), [bottom_pad right_pad], avg_chans(1), 'post');
        G = padarray(im(:,:,2), [bottom_pad right_pad], avg_chans(2), 'post');
        B = padarray(im(:,:,3), [bottom_pad right_pad], avg_chans(3), 'post');
        im = cat(3, R, G, B);
    end

    xs = context_xmin : context_xmax;
    ys = context_ymin : context_ymax;

    im_patch_original = im(ys, xs, :);
    if ~isequal(model_sz, original_sz)
        im_patch = imresize(im_patch_original, model_sz);
    else
        im_patch = im_patch_original;
    end
end

