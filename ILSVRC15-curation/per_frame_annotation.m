% --------------------------------------------------------------------------
function per_frame_annotation(root, folder_index)
% Read per-frame XML annotations and write bbox and track info on txt files
% Argument is folder_index [0-4]
% e.g. per_frame_annotations('/path/to/ILSVRC2015/', 0)
% --------------------------------------------------------------------------

    addpath(genpath('../..'));
    i = folder_index + 3;

    CLASS_IDS = {'n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808', ...
                     'n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045', ...
                     'n02509815','n02411705','n01726692','n02355227','n02129604','n04468005','n01662784','n04530566','n02062744','n02391049'};

%     CLASS_NAMES = {'airplane','antelope','bear','bicycle','bird','bus','car','cattle','dog','domestic_cat', ...
%                    'elephant','fox','giant_panda','hamster','horse','lion', 'lizard','monkey','motorcycle','rabbit', ...
%                    'red_panda','sheep','snake','squirrel','tiger','train','turtle','watercraft','whale','zebra'};

    OLD_FOLDERS = {'ILSVRC2015_VID_train_0000','ILSVRC2015_VID_train_0001','ILSVRC2015_VID_train_0002','ILSVRC2015_VID_train_0004','val'};
    NEW_FOLDERS = {'a','b','c','d','e'};

    root_anno = [root 'Annotations/VID/train'];
    root_imgs = [root 'Data/VID/train'];
    slash = '/';

    dir_root_anno = dir(root_anno);
    dir_root_anno = {dir_root_anno.name};
    % we always start from 3 because at 1 we have '.' and at 2 '..'
    level1 = [root_anno slash dir_root_anno{i}];
    dir_level1 = dir(level1);
    dir_level1 = {dir_level1.name};
    % j iterates across videos
    for j=3:numel(dir_level1)
        level2 = [root_anno slash dir_root_anno{i} slash dir_level1{j}];
        dir_level2 = dir(level2);
        dir_level2 = {dir_level2.name};
        xml_path = [root_anno slash dir_root_anno{i} slash dir_level1{j} slash dir_level2{3}];
        struct_from_xml = xml2struct_custom(xml_path);
        % get frame size from first frame of the video
        frame_sz = [str2double(struct_from_xml.annotation.size.height.Text), str2double(struct_from_xml.annotation.size.width.Text)];

        % k iterates across frames
        for k=3:numel(dir_level2)
            fprintf('Processing i: %d j: %d k: %d\n', i-3, j-3, k-3);
            xml_path = [root_anno slash dir_root_anno{i} slash dir_level1{j} slash dir_level2{k}];
            struct_from_xml = xml2struct_custom(xml_path);
            dir_level2_img = strrep(dir_level2{k}, '.xml', '.JPEG');
            im_path = [root_imgs slash dir_root_anno{i} slash dir_level1{j} slash dir_level2_img];
            im_anno_path = strrep(im_path, '.JPEG', '.txt');
            fid = fopen(im_anno_path,'w');
            % check if the frame contains objects at all
            if isfield(struct_from_xml.annotation, 'object')
                n_objects = numel(struct_from_xml.annotation.object);
                % k iterates across objects
                for o=1:n_objects
                    if n_objects==1
                        o_class = struct_from_xml.annotation.object(o).name.Text;
                        o_xmax = str2double(struct_from_xml.annotation.object(o).bndbox.xmax.Text);
                        o_xmin = str2double(struct_from_xml.annotation.object(o).bndbox.xmin.Text);
                        o_ymax = str2double(struct_from_xml.annotation.object(o).bndbox.ymax.Text);
                        o_ymin = str2double(struct_from_xml.annotation.object(o).bndbox.ymin.Text);
                        trackid = str2double(struct_from_xml.annotation.object(o).trackid.Text);
                    else
                        o_class = struct_from_xml.annotation.object{o}.name.Text;
                        o_xmax = str2double(struct_from_xml.annotation.object{o}.bndbox.xmax.Text);
                        o_xmin = str2double(struct_from_xml.annotation.object{o}.bndbox.xmin.Text);
                        o_ymax = str2double(struct_from_xml.annotation.object{o}.bndbox.ymax.Text);
                        o_ymin = str2double(struct_from_xml.annotation.object{o}.bndbox.ymin.Text);
                        trackid = str2double(struct_from_xml.annotation.object{o}.trackid.Text);
                    end
                    % convert ImageNet WNID of the class to integer [1-30]
                    [~, o_class] = ismember(o_class, CLASS_IDS);
                    o_sz = [o_ymax - o_ymin + 1 , o_xmax - o_xmin + 1];

                    fprintf(fid, '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n', ...
                        i, j, k, o, trackid, o_class, ...
                        frame_sz(2), frame_sz(1),  ...
                        o_xmin, o_ymin, o_sz(2), o_sz(1), im_path);
                end
            else
                    fprintf(fid, '%d,%d,%d,%d,%d,%s\n', ...
                        i, j, k, ...
                        frame_sz(2), frame_sz(1), im_path);
            end
            fclose(fid);
        end
    end
end
