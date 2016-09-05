% ----------------------------------------------------------------------------------------------------------------
function net = vid_create_net(varargin)
% Very similar to vanilla AlexNet from MatConvNet examples,
% but with smaller stride at conv1 and no padding
% Used to generate the network described in  the paper
% "Fully-Convolutional Siamese Networks for Object Tracking"
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
    opts.strides = [2, 2, 1, 2] ;
    opts.cudnnWorkspaceLimit = 1024*1024*1024 ; % 1GB
    opts = vl_argparse(opts, varargin) ;

    if numel(opts.exemplarSize) == 1
        opts.exemplarSize = [opts.exemplarSize, opts.exemplarSize];
    end
    if numel(opts.instanceSize) == 1
        opts.instanceSize = [opts.instanceSize, opts.instanceSize];
    end

    net = modified_alexnet(struct(), opts) ;

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
function net = modified_alexnet(net, opts)
% --------------------------------------------------------------------

    strides = ones(1, 7);
    strides(1:numel(opts.strides)) = opts.strides(:);

    net.layers = {} ;

    net = add_block(net, opts, '1', 11, 11, 3, 96, strides(1), 0) ;
    net = add_norm(net, opts, '1') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', strides(2), ...
                               'pad', 0) ;

    net = add_block(net, opts, '2', 5, 5, 48, 256, strides(3), 0) ;
    net = add_norm(net, opts, '2') ;
    net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', strides(4), ...
                               'pad', 0) ;

    net = add_block(net, opts, '3', 3, 3, 256, 384, strides(5), 0) ;
    net = add_block(net, opts, '4', 3, 3, 192, 384, strides(6), 0) ;
    net = add_block_conv_only(net, opts, '5', 3, 3, 192, 256, strides(7), 0) ;

    % Check if the receptive field covers full image

    [ideal_exemplar, ~] = ideal_size(net, opts.exemplarSize);
    [ideal_instance, ~] = ideal_size(net, opts.instanceSize);
    assert(sum(opts.exemplarSize==ideal_exemplar)==2, 'exemplarSize is not ideal.');
    assert(sum(opts.instanceSize==ideal_instance)==2, 'instanceSize is not ideal.');
end
