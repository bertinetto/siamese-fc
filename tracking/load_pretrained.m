% -------------------------------------------------------------------------------------------------
function net = load_pretrained(netPath, gpu)
%LOAD_PRETRAINED loads a pretrained fully-convolutional Siamese network as a DagNN
% -------------------------------------------------------------------------------------------------
    % to keep consistency when reading all hyperparams
    if iscell(netPath)
        netPath = netPath{1};
    end
    trainResults = load(netPath);
    net = trainResults.net;
    % remove legacy fields if present
    [~,xcorrId] = find_layers_from_type(net, 'XCorr');
    xcorrId = xcorrId{1};
    if isfield(net.layers(xcorrId).block, 'expect')
        net.layers(xcorrId).block = rmfield(net.layers(xcorrId).block,'expect');
    end
    if isfield(net.layers(xcorrId).block, 'visualization_active')
        net.layers(xcorrId).block = rmfield(net.layers(xcorrId).block,'visualization_active');
    end
    if isfield(net.layers(xcorrId).block, 'visualization_grid_sz')
        net.layers(xcorrId).block = rmfield(net.layers(xcorrId).block,'visualization_grid_sz');
    end

    % load as DAGNN
    net = dagnn.DagNN.loadobj(net);
    % remove loss layer
    net = remove_layers_from_block(net, 'dagnn.Loss');
    % init specified GPU
    if ~isempty(gpu)
       gpuDevice(gpu)
    end
    net.move('gpu');
    net.mode = 'test'; % very important for batch norm, we now use the stats accumulated during training.
end