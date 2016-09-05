% -------------------------------------------------------------------------------------------------
function net = make_siameseFC(opts)
%MAKE_SIAMESEFC
%   Creates Siamese Fully-Convolutional network,
%   made by duplicating a vanilla AlexNet in two branches
%   and joining the branches with a cross-correlation layer
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------
    net_opts.siamese = true;
    net_opts.batchNormalization = true;
    net_opts.strides = [2, 2, 1, 2];
    net_opts = vl_argparse(net_opts, {opts.net.conf});

    if opts.pretrain
        error('not yet implemented for alexnet');
    end

    % create siamese branch of the network (5 conv layers)
    fprintf('construct network\n');
    % four different net sizes to tradeoff accuracy/speed: vid_create_net, _small1, _small2 and _small3
    branch = vid_create_net(...
        'exemplarSize',       opts.exemplarSize * [1 1], ...
        'instanceSize',       opts.instanceSize * [1 1], ...
        'batchNormalization', net_opts.batchNormalization, ...
        'networkType',        'simplenn', ...
        'weightInitMethod',   opts.init.weightInitMethod, ...
        'scale',              opts.init.scale, ...
        'initBias',           opts.init.initBias, ...
        'strides',            net_opts.strides);

    branch = dagnn.DagNN.fromSimpleNN(branch);

    % Add common final stream of the network
    net = dagnn.DagNN();
    rename_io_vars(branch, 'in', 'out');
    add_pair_of_streams(net, branch, ...
                        {'exemplar', 'instance'}, ...
                        {'a_feat', 'b_feat'}, ...
                        net_opts.siamese);

    net.addLayer('xcorr', XCorr(), ...
                 {'a_feat', 'b_feat'}, ...
                 {'xcorr_out'}, ...
                 {});

    add_adjust_layer(net, 'adjust', 'xcorr_out', 'score', ...
                 {'adjust_f', 'adjust_b'}, 1e-3, 0, 0, 1);

end
