% -------------------------------------------------------------------------------------------------------------------------
function net = add_dag_to_dag(net, orig, rename)
%ADD_DAG_TO_DAG
%   Copies one DAG into another.
%   The blocks of all layers are copied.
%   The following fields of params are copied:
%   value, trainMethod, learningRate, weightDecay
%
%   The optional rename parameter can be empty, a function handle or a struct.
%   If rename is empty, all names are preserved.
%   If rename is a function handle (string -> string),
%   then it is used to rename layers, vars and params.
%   Otherwise, rename is a struct with properties
%   rename.layer (func : string -> string)
%   rename.var (func : string -> string)
%   rename.param (func : string -> string)
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------


if nargin < 3
    rename = [];
end
if ~isempty(rename)
    if isa(rename, 'function_handle')
        rename = struct('layer', rename, 'var', rename, 'param', rename);
    end
end

% Copy blocks.
for layer = orig.layers
    name = layer.name;
    inputs = layer.inputs;
    outputs = layer.outputs;
    params = layer.params;
    if ~isempty(rename)
        name = rename.layer(name);
        inputs = cellfun(rename.var, inputs, 'UniformOutput', false);
        outputs = cellfun(rename.var, outputs, 'UniformOutput', false);
        params = cellfun(rename.param, params, 'UniformOutput', false);
    end
    net.addLayer(name, layer.block, inputs, outputs, params);
end

% Get list of params that were copied.
all_params = {};
for layer = orig.layers
    all_params = [all_params, layer.params];
end
all_params = unique(all_params);

% Copy properties of each param.
for i = 1:numel(all_params)
    orig_name = all_params{i};
    orig_ind = orig.getParamIndex(orig_name);
    name = orig_name;
    if ~isempty(rename)
        name = rename.param(name);
    end
    ind = net.getParamIndex(name);
    net.params(ind).value        = orig.params(orig_ind).value;
    net.params(ind).trainMethod  = orig.params(orig_ind).trainMethod;
    net.params(ind).learningRate = orig.params(orig_ind).learningRate;
    net.params(ind).weightDecay  = orig.params(orig_ind).weightDecay;
end

end
