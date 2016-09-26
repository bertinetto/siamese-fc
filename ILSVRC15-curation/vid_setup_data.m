% ------------------------------------------------------------------------
function imdb_video = vid_setup_data(root)
    %VID_SETUP_DATA
    % creates an IMDB structure pointing to the data
	% e.g. vid_setup_data('/path/to/ILSVRC15/')
% ------------------------------------------------------------------------
    rootpath = [root 'Data/VID/train/'];
    MAX_TRACKIDS = 50; % upper bound for num of objects in the same frame
    frames_id_path = './vid_id_frames.txt';
    imdb_video = struct();
    vf = fopen(frames_id_path);
    % first col: ids, second col: nframes, third col: folder
    video_info = textscan(vf,'%s %d %d', 'Delimiter', ' ');
    fclose(vf);

    video_paths = video_info{1};
    video_ids = video_info{2};
    video_nframes = video_info{3};
    n_videos = size(video_ids,1);
    imdb_video.id = uint32(video_ids'); % relates to the full imdb
    imdb_video.path = video_paths';
    imdb_video.nframes = uint32(video_nframes)';
    imdb_video.n_valid_objects = uint32(zeros(1, n_videos));
    imdb_video.valid_trackids = uint16(zeros(MAX_TRACKIDS, n_videos));
    imdb_video.valid_per_trackid = cell(MAX_TRACKIDS, n_videos);
    imdb_video.total_valid_objects = uint32(0);
    imdb_video.objects = cell(1, n_videos);

    for v=1:numel(imdb_video.id)
        fprintf('Objects from video %d/%d ...\t', v, n_videos);
        obj_structs = struct('frame_path', {}, 'track_id', {}, 'extent', {}, 'frame_sz', {}, 'class', {}, 'valid', {});
        % open video file
        fv = fopen([rootpath imdb_video.path{v} '.txt'], 'r');
        l = 1;
        line = fgetl(fv);
        while ischar(line)
            % reading trackid, class, frame_w, frame_h, o_xmins, o_ymins, o_ws, hs, frame_path
            V = strsplit(line,',');
            obj_structs{l}.track_id = uint8(str2double(V{1}));
            obj_structs{l}.class = uint8(str2double(V{2}));
            obj_structs{l}.frame_sz = uint16([str2double(V{3}) str2double(V{4})]);
            obj_structs{l}.extent = int16([str2double(V{5}) str2double(V{6}) str2double(V{7}) str2double(V{8})]);
            % here we can decide contraints for the validity of an object
            obj_structs{l}.valid = true;
            % obj_structs{l}.valid = checkClass(obj_structs{l}.class) && ...
                                            % checkSize(obj_structs{l}.frame_sz, obj_structs{l}.extent) && ...
                                            % checkBorders(obj_structs{l}.frame_sz, obj_structs{l}.extent);
            full_path = V{9};
            % discard the first fixed part of the path
            ss = strsplit(full_path,'train/');
            obj_structs{l}.frame_path = ss{2};
            if obj_structs{l}.valid
                % update count of valid objects and trackids
                imdb_video.n_valid_objects(v) = imdb_video.n_valid_objects(v) + 1;
                % trackids start from 0, but indexes start from 1
                imdb_video.valid_trackids(obj_structs{l}.track_id+1, v) = imdb_video.valid_trackids(obj_structs{l}.track_id+1, v) + 1;
                % save list of valid objects per trackid for easy random choice pairs
                imdb_video.valid_per_trackid{obj_structs{l}.track_id+1, v}(end+1) = uint16(l);
            end
            l = l+1;
            line = fgetl(fv);
        end
        fclose(fv);
        imdb_video.objects{v} = obj_structs;
        imdb_video.total_valid_objects = imdb_video.total_valid_objects + imdb_video.n_valid_objects(v);
        fprintf('Found %d valid objects in %d frames\n', imdb_video.n_valid_objects(v), imdb_video.nframes(v));
    end
    % to_delete: videos which contains trackids with only one valid object
    to_delete1 = find(imdb_video.n_valid_objects < 2);
    imdb_video = delete_from_imdb(imdb_video, to_delete1);
    % to_delete: videos which only contains trackids with only one valid object
    [~, to_delete2] = find(imdb_video.valid_trackids==1);
    to_delete2 = unique(to_delete2);
    imdb_video = delete_from_imdb(imdb_video, to_delete2);
    save('imdb_video.mat', '-struct', 'imdb_video', '-v7.3');
end

%% Not used at the moment
% function ok = checkClass(class)
%     forbidden_classes = {'lizard','snake','train','whale'};
%     ok = ~ismember(class, forbidden_classes);
% end

% function ok = checkSize(frame_sz, object_extent)
%     min_ratio = 0.1;
%     max_ratio = 0.75;
%     % accept only objects >10% and <75% of the total frame
%     area_ratio = sqrt((object_extent(3)*object_extent(4))/prod(frame_sz));
%     ok = area_ratio > min_ratio && area_ratio < max_ratio;
% end

% function ok = checkBorders(frame_sz, object_extent)
%     dist_from_border = 0.05 * (object_extent(3) + object_extent(4))/2;
%     ok = object_extent(1) > dist_from_border && object_extent(2) > dist_from_border && ...
%                 (frame_sz(1)-(object_extent(1)+object_extent(3))) > dist_from_border && ...
%                 (frame_sz(2)-(object_extent(2)+object_extent(4))) > dist_from_border;
% end

% -----------------------------------------------------------------------------------------------
function imdb = delete_from_imdb(imdb, to_delete)
% -----------------------------------------------------------------------------------------------
    imdb.total_valid_objects =  imdb.total_valid_objects - sum(imdb.n_valid_objects(to_delete));
    imdb.id(to_delete) = [];
    imdb.path(to_delete) = [];
    imdb.nframes(to_delete) = [];
%     imdb.set(to_delete) = [];
    imdb.n_valid_objects(to_delete) = [];
    imdb.objects(to_delete) = [];
    imdb.valid_trackids(:,to_delete) = [];
    imdb.valid_per_trackid(:, to_delete) = [];
end

