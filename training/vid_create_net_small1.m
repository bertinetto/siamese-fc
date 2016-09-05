% ----------------------------------------------------------------------------------------------------------------
function [net, fixed_label_size] = vid_create_net_small1(varargin)
% Very similar to vanilla AlexNet from MatConvNet examples,
% but with smaller stride at conv1 and no padding
% The paper "Fully-Convolutional Siamese Networks for Object Tracking" uses vid_create_net_small,
% this is a smaller version which performs slightly worse but runs faster for both training and tracking.
% ----------------------------------------------------------------------------------------------------------------
    opts.exemplarSize = [127 127];
    opts.instanceSize = [255 255];
    opts.scale = 1 ;
    opts.initBias = 0.1 ;
    opts.weightDecay = 1 ;
    %opts.weightInitMethod = 'xavierimproved' ;
    opts.weightInitMethod = 'gaussian';
    opts.batchNormalization = false ;
    opts.networkType = 'simplenn' ;
    opts.cudnnWorkspaceLimit = 1024*1024*1024 ; % 1GB
    opts = vl_argparse(opts, varargin) ;

    [net, fixed_label_size] = modified_alexnet(struct(), opts) ;

    % Meta parameters
    net.meta.normalization.interpolation = 'bicubic' ;
    net.meta.normalization.averageImage = [] ;
    net.meta.normalization.keepAspect = true ;
    net.meta.augmentation.rgbVariance = zeros(0,3) ;
    net.meta.augmentation.transformation = 'stretch' ;

    % Fill in default values
    net = vl_simplenn_tidy(net) ;

    % Switch to DagNN if requested
    switch lower(opts.networkType)
      case 'simplenn'
        % done
      case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
                     {'prediction','label'}, 'top1err') ;
        net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
                                           'opts', {'topK',5}), ...
                     {'prediction','label'}, 'top5err') ;
      otherwise
        assert(false) ;
    end

end

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end

% --------------------------------------------------------------------
function net = add_block_conv_only(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end
end

% --------------------------------------------------------------------
function [net, fixed_label_size] = modified_alexnet(net, opts)
% --------------------------------------------------------------------

    net.layers = {} ;

    %% smaller filters, all /2
    net = add_block(net, opts, '1', 11, 11, 3, 48, 2, 0) ;
    net = add_norm(net, opts, '1') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = add_block(net, opts, '2', 5, 5, 48, 128, 1, 0) ;
    net = add_norm(net, opts, '2') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', 0) ;

    net = add_block(net, opts, '3', 3, 3, 128, 192, 1, 0) ;
    net = add_block(net, opts, '4', 3, 3, 192, 192, 1, 0) ;
    net = add_block_conv_only(net, opts, '5', 3, 3, 192, 128, 1, 0) ;

    % Check if the receptive field covers full image

    [ideal_exemplar, ~] = ideal_size(net, opts.exemplarSize);
    [ideal_instance, ~] = ideal_size(net, opts.instanceSize);
    assert(sum(opts.exemplarSize==ideal_exemplar)==2, 'exemplarSize is not ideal.');
    assert(sum(opts.instanceSize==ideal_instance)==2, 'instanceSize is not ideal.');
    netsize_exemplar = vl_simplenn_display(net, 'inputSize', [opts.exemplarSize 3 8]);
    netsize_instance = vl_simplenn_display(net, 'inputSize', [opts.instanceSize 3 8]);

    exemplar_final_size = netsize_exemplar.dataSize(1:2, end);
    instance_final_size = netsize_instance.dataSize(1:2, end);
    fixed_label_size = instance_final_size - exemplar_final_size + 1;
    % We want an odd number so that we can center the target in the middle
    assert(mod(fixed_label_size(1),2)~=0, 'Fixed label size should be an odd number.');
end