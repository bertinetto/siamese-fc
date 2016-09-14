function net = make_siamese(stream1, stream2, join, final, inputs, output, varargin)
% Constructs a Siamese network of two stream nets joined by the join and
% then followed by the final net.
% The stream and final nets can be simple or DAG.
% They should have one input and one output.
% The two streams should be identical except for parameter values.
% The join net is a DAG with 2 inputs and 1 output.
% The inputs must be named 'in1' and 'in2'.
% The inputs and output parameters are the variable names for the resulting net.
% The input names is a cell-array of 2 strings, the output names is a string.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
opts.share_all = true;
% List of params to share, or layers whose params should be shared.
% Ignored if share_all is true.
opts.share_params = [];
opts.share_layers = [];
opts = vl_argparse(opts, varargin) ;

stream_outputs = {'br1_out', 'br2_out'};
join_output = 'join_out';

% Assume that indices of layers are preserved for share_layers.
if ~isa(stream1, 'dagnn.DagNN')
    stream1 = dagnn.DagNN.fromSimpleNN(stream1);
end
if ~isa(stream2, 'dagnn.DagNN')
    stream2 = dagnn.DagNN.fromSimpleNN(stream2);
end
if ~isempty(final)
    if ~isa(final, 'dagnn.DagNN')
        final = dagnn.DagNN.fromSimpleNN(final);
    end
end
if isempty(final)
    join_output = output;
end

if opts.share_all
    opts.share_params = 1:numel(stream1.params);
else
    % Find all params that belong to share_layers.
    opts.share_params = union(opts.share_params, ...
                              params_of_layers(stream1, opts.share_layers));
end

net = dagnn.DagNN();
add_branches(net, stream1, stream2, inputs, stream_outputs, opts.share_params);
add_join(net, join, {'in1', 'in2'}, stream_outputs, join_output)
if ~isempty(final)
    add_final(net, final, join_output, output)
end

end

function add_branches(net, stream1, stream2, inputs, outputs, share_inds)
% share_inds is a list of params to share.

    % Assume that both streams have the same names.
    orig_input = only(stream1.getInputs());
    orig_output = only(stream1.getOutputs());
    % Convert param indices to names.
    share_names = arrayfun(@(l) l.name, stream1.params(share_inds), ...
                           'UniformOutput', false);

    rename_unique1 = @(s) ['br1_', s];
    rename_unique2 = @(s) ['br2_', s];
    rename_common = @(s) ['br_', s];
    rename1 = struct(...
        'layer', rename_unique1, ...
        'var', rename_unique1, ...
        'param', @(s) rename_pred(s, @(x) ismember(x, share_names), ...
                                  rename_common, rename_unique1));
    rename2 = struct(...
        'layer', rename_unique2, ...
        'var', rename_unique2, ...
        'param', @(s) rename_pred(s, @(x) ismember(x, share_names), ...
                                  rename_common, rename_unique2));

    add_dag_to_dag(net, stream1, rename1);
    % Values of shared params will be taken from stream2
    % since add_dag_to_dag over-writes existing parameters.
    add_dag_to_dag(net, stream2, rename2);
    net.renameVar(rename_unique1(orig_input), inputs{1});
    net.renameVar(rename_unique2(orig_input), inputs{2});
    net.renameVar(rename_unique1(orig_output), outputs{1});
    net.renameVar(rename_unique2(orig_output), outputs{2});
end

function r = rename_pred(s, pred, rename_true, rename_false)
    if pred(s)
        r = rename_true(s);
    else
        r = rename_false(s);
    end
end

function add_join(net, join, orig_inputs, inputs, output)
    assert(numel(join.getInputs()) == 2);
    orig_output = only(join.getOutputs());

    rename_join = @(s) ['join_', s];
    add_dag_to_dag(net, join, rename_join);
    for i = 1:2
        net.renameVar(rename_join(orig_inputs{i}), inputs{i});
    end
    net.renameVar(rename_join(orig_output), output);
end

function add_final(net, final, input, output)
    orig_inputs = final.getInputs();
    orig_outputs = final.getOutputs();
    assert(numel(orig_inputs) == 1);
    assert(numel(orig_outputs) == 1);
    orig_input = orig_inputs{1};
    orig_output = orig_outputs{1};

    rename_final = @(s) ['fin_', s];
    add_dag_to_dag(net, final, rename_final);
    net.renameVar(rename_final(orig_input), input);
    net.renameVar(rename_final(orig_output), output);
end

function param_inds = params_of_layers(net, layer_inds)
    layer_params = arrayfun(@(l) l.params, net.layers(layer_inds), ...
                            'UniformOutput', false);
    param_names = cat(2, {}, layer_params{:});
    param_inds = cellfun(@(s) net.getParamIndex(s), param_names);
    param_inds = unique(param_inds);
end
