% -------------------------------------------------------------------------------------------------
function [net, stats] = experiment(imdb_video, varargin)
%EXPERIMENT
%   main function - creates a network and trains it on the dataset indexed by imdb_video.
%
%   Luca Bertinetto, Jack Valmadre, Joao Henriques, 2016
% -------------------------------------------------------------------------------------------------
    % Default parameters (set the experiment-specific ones in run_experiment)
    opts.net.type = 'alexnet';
    opts.net.conf = struct(); % Options depend on type of net.
    opts.pretrain = false; % Location of model file set in env_paths.
    opts.init.scale = 1;
    opts.init.weightInitMethod = 'xavierimproved';
    opts.init.initBias = 0.1;
    opts.expDir = 'data'; % where to save the trained net
    opts.numFetchThreads = 12; % used by vl_imreadjpg when reading dataset
    opts.validation = 0.1; % fraction of imbd reserved to validation
    opts.exemplarSize = 127; % exemplar (z) in the paper
    opts.instanceSize = 255 - 2*8; % search region (x) in the paper
    opts.loss.type = 'simple';
    opts.loss.rPos = 16; % pixel with distance from center d > rPos are given a negative label
    opts.loss.rNeg = 0; % if rNeg != 0 pixels rPos < d < rNeg are given a neutral label
    opts.loss.labelWeight = 'balanced';
    opts.numPairs =  5.32e4; % Number of example pairs per epoch, if empty, then equal to number of videos.
    opts.randomSeed = 0;
    opts.shuffleDataset = false; % do not shuffle the data to get reproducible experiments
    opts.frameRange = 100; % range from the exemplar in which randomly pick the instance
    opts.gpus = [];
    opts.prefetch = false; % Both get_batch and cnn_train_dag depend on prefetch.
    opts.train.numEpochs = 50;
    opts.train.learningRate = logspace(-2, -5, opts.train.numEpochs);
    opts.train.weightDecay = 5e-4;
    opts.train.batchSize = 8; % we empirically observed that small batches work better
    opts.train.profile = false;
    % Data augmentation settings
    opts.subMean = false;
    opts.colorRange = 255;
    opts.augment.translate = true;
    opts.augment.maxTranslate = 4;
    opts.augment.stretch = true;
    opts.augment.maxStretch = 0.05;
    opts.augment.color = true;
    opts.augment.grayscale = 0; % likelihood of using grayscale pair
    % Override default parameters if specified in run_experiment
    opts = vl_argparse(opts, varargin);
    % Get environment-specific default paths.
    opts = env_paths_training(opts);
    opts.train.gpus = opts.gpus;
    opts.train.prefetch = opts.prefetch;
% -------------------------------------------------------------------------------------------------
    % Get ImageNet Video metadata
    if isempty(imdb_video)
        fprintf('loading imdb video...\n');
        imdb_video = load(opts.imdbVideoPath);
    end

    % Load dataset statistics
    [rgbMean_z, rgbVariance_z, rgbMean_x, rgbVariance_x] = load_stats(opts);
    if opts.shuffleDataset
        s = RandStream.create('mt19937ar', 'Seed', 'shuffle');
        opts.randomSeed = s.Seed;
    end

    opts.train.expDir = opts.expDir;

    rng(opts.randomSeed); % Re-seed before calling make_net.

    % -------------------------------------------------------------------------------------------------
    net = make_net(opts);
    % -------------------------------------------------------------------------------------------------

    [imdb_video, imdb] = choose_val_set(imdb_video, opts);

    [resp_sz, resp_stride] = get_response_size(net, opts);
    % We want an odd number so that we can center the target in the middle
    assert(all(mod(resp_sz, 2) == 1), 'resp. size is not odd');

    [net, derOutputs, label_inputs_fn] = setup_loss(net, resp_sz, resp_stride, opts.loss);

    batch_fn = @(db, batch) get_batch(db, batch, ...
                                        imdb_video, ...
                                        opts.rootDataDir, ...
                                        numel(opts.train.gpus) >= 1, ...
                                        struct('exemplarSize', opts.exemplarSize, ...
                                               'instanceSize', opts.instanceSize, ...
                                               'frameRange', opts.frameRange, ...
                                               'subMean', opts.subMean, ...
                                               'colorRange', opts.colorRange, ...
                                               'stats', struct('rgbMean_z', rgbMean_z, ...
                                                               'rgbVariance_z', rgbVariance_z, ...
                                                               'rgbMean_x', rgbMean_x, ...
                                                               'rgbVariance_x', rgbVariance_x), ...
                                               'augment', opts.augment, ...
                                               'prefetch', opts.train.prefetch, ...
                                               'numThreads', opts.numFetchThreads), ...
                                        label_inputs_fn);

    opts.train.derOutputs = derOutputs;
    opts.train.randomSeed = opts.randomSeed;
    % -------------------------------------------------------------------------------------------------
    % Start training
    [net, stats] = cnn_train_dag(net, imdb, batch_fn, opts.train);
    % -------------------------------------------------------------------------------------------------
end


% -----------------------------------------------------------------------------------------------------
function [rgbMean_z, rgbVariance_z, rgbMean_x, rgbVariance_x] = load_stats(opts)
% Dataset image statistics for data augmentation
% -----------------------------------------------------------------------------------------------------
    stats = load(opts.imageStatsPath);
    % Subtracted if opts.subMean is true
    if ~isfield(stats, 'z')
        rgbMean = reshape(stats.rgbMean, [1 1 3]);
        rgbMean_z = rgbMean;
        rgbMean_x = rgbMean;
        [v,d] = eig(stats.rgbCovariance);
        rgbVariance_z = 0.1*sqrt(d)*v';
        rgbVariance_x = 0.1*sqrt(d)*v';
    else
        rgbMean_z = reshape(stats.z.rgbMean, [1 1 3]);
        rgbMean_x = reshape(stats.x.rgbMean, [1 1 3]);
        % Set data augmentation statistics, used if opts.augment.color is true
        [v,d] = eig(stats.z.rgbCovariance);
        rgbVariance_z = 0.1*sqrt(d)*v';
        [v,d] = eig(stats.x.rgbCovariance);
        rgbVariance_x = 0.1*sqrt(d)*v';
    end
end


% -------------------------------------------------------------------------------------------------
function net = make_net(opts)
% -------------------------------------------------------------------------------------------------

    net = make_siameseFC(opts);

    % Save the net graph to disk.
    inputs = {'exemplar', [opts.exemplarSize*[1 1] 3 opts.train.batchSize], ...
              'instance', [opts.instanceSize*[1 1] 3 opts.train.batchSize]};
    net_dot = net.print(inputs, 'Format', 'dot');
    if ~exist(opts.expDir)
        mkdir(opts.expDir);
    end
    f = fopen(fullfile(opts.expDir, 'arch.dot'), 'w');
    fprintf(f, net_dot);
    fclose(f);
end


% -------------------------------------------------------------------------------------------------
function [resp_sz, resp_stride] = get_response_size(net, opts)
% -------------------------------------------------------------------------------------------------

    sizes = net.getVarSizes({'exemplar', [opts.exemplarSize*[1 1] 3 256], ...
                             'instance', [opts.instanceSize*[1 1] 3 256]});
    resp_sz = sizes{net.getVarIndex('score')}(1:2);
    rfs = net.getVarReceptiveFields('exemplar');
    resp_stride = rfs(net.getVarIndex('score')).stride(1);
    assert(all(rfs(net.getVarIndex('score')).stride == resp_stride));
end


% -------------------------------------------------------------------------------------------------
function [net, derOutputs, inputs_fn] = setup_loss(net, resp_sz, resp_stride, loss_opts)
% Add layers to the network, specifies the losses to minimise, and
% constructs a function that returns the inputs required by the loss layers.
% -------------------------------------------------------------------------------------------------

    % create label and weights for logistic loss
    net.addLayer('objective', ...
                 dagnn.Loss('loss', 'logistic'), ...
                 {'score', 'eltwise_label'}, 'objective');
    % adding weights to loss layer
    [pos_eltwise, instanceWeight] = create_labels(...
        resp_sz, loss_opts.labelWeight, ...
        loss_opts.rPos/resp_stride, loss_opts.rNeg/resp_stride);
    neg_eltwise = [];   % no negative pairs at the moment
    net.layers(end).block.opts = [...
        net.layers(end).block.opts, ...
        {'instanceWeights', instanceWeight}];

    derOutputs = {'objective', 1};

    inputs_fn = @(labels, obj_sz_z, obj_sz_x) get_label_inputs_simple(...
        labels, obj_sz_z, obj_sz_x, pos_eltwise, neg_eltwise);


    net.addLayer('errdisp', centerThrErr(), {'score', 'label'}, 'errdisp');
    net.addLayer('errmax', MaxScoreErr(), {'score', 'label'}, 'errmax');
end


% -------------------------------------------------------------------------------------------------
function inputs = get_label_inputs_simple(labels, obj_sz_z, obj_sz_x, pos_eltwise, neg_eltwise)
% GET_LABEL_INPUTS_SIMPME returns the network inputs that specify the labels.
%
% labels -- Label of +1 or -1 per image pair, size [1, n].
% obj_sz_z -- Size of exemplar box, dims [2, n].
% obj_sz_x -- Size of instance box, dims [2, n].
% -------------------------------------------------------------------------------------------------

    pos = (labels > 0);
    neg = (labels < 0);

    resp_sz = size(pos_eltwise);
    eltwise_labels = zeros([resp_sz, 1, numel(labels)], 'single');
    eltwise_labels(:,:,:,pos) = repmat(pos_eltwise, [1 1 1 sum(pos)]);
    eltwise_labels(:,:,:,neg) = repmat(neg_eltwise, [1 1 1 sum(neg)]);
    inputs = {'label', labels, ...
              'eltwise_label', eltwise_labels};
end


% -------------------------------------------------------------------------------------------------
function [imdb_video, imdb] = choose_val_set(imdb_video, opts)
% Designates some examples for validation.
% It modifies imdb_video and constructs a dummy imdb.
% -------------------------------------------------------------------------------------------------
    TRAIN_SET = 1;
    VAL_SET = 2;

    % set opts.validation to validation and the rest to training.
    size_dataset = numel(imdb_video.id);
    size_validation = round(opts.validation * size_dataset);
    size_training = size_dataset - size_validation;
    imdb_video.set = uint8(zeros(1, size_dataset));
    imdb_video.set(1:size_training) = TRAIN_SET;
    imdb_video.set(size_training+1:end) = VAL_SET;

    %% create imdb of indexes to imdb_video
    % train and val from disjoint video sets
    imdb = struct();
    imdb.images = struct(); % we keep the images struct for consistency with cnn_train_dag (MatConvNet)
    imdb.id = 1:opts.numPairs;
    n_pairs_train = round(opts.numPairs * (1-opts.validation));
    imdb.images.set = uint8(zeros(1, opts.numPairs)); % 1 -> train
    imdb.images.set(1:n_pairs_train) = TRAIN_SET;
    imdb.images.set(n_pairs_train+1:end) = VAL_SET;
end


% -------------------------------------------------------------------------------------------------
function inputs = get_batch(db, batch, imdb_video, data_dir, use_gpu, sample_opts, label_inputs_fn)
% Returns the inputs to the network.
% -------------------------------------------------------------------------------------------------

    [imout_z, imout_x, labels, sizes_z, sizes_x] = vid_get_random_batch(...
        db, imdb_video, batch, data_dir, sample_opts);
    if use_gpu
        imout_z = gpuArray(imout_z);
        imout_x = gpuArray(imout_x);
    end
    % Constructs full label inputs from output of vid_get_random_batch.
    label_inputs = label_inputs_fn(labels, sizes_z, sizes_x);
    inputs = [{'exemplar', imout_z, 'instance', imout_x}, label_inputs];
end
