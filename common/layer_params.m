% -------------------------------------------------------------------------------------------------------------------------
function params = layer_params(net, layers)
% -------------------------------------------------------------------------------------------------------------------------
    % layers are layer indices.
    % params are param names.
    params = arrayfun(@(i) net.layers(i).params, layers, 'UniformOutput', false);
    params = cat(2, params{:});
    params = unique(params);
end