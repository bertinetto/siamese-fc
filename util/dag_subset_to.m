% -------------------------------------------------------------------------------------------------------------------------
function subset = dag_subset_to(net, nodes)
%DAG_SUBSET_TO
% Finds the layers in a net that the nodes depend on.
% (The ancestors of nodes in the DAG.)
% nodes is a cell array of layer names.
% Returns a list of layer indices.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------

subset = [];
for i = 1:numel(nodes)
    s = net.getLayerIndex(nodes{i});
    if isnan(s)
        error(sprintf('layer not found: %s', nodes{i}));
    end
    subset = [subset, subset_to_one(net, s)];
end
subset = unique(subset);

end

function subset = subset_to_one(net, l)
    % l is a layer index.
    vars = net.layers(l).inputs;
    subset = [l];
    for i = 1:numel(vars)
        var = vars{i};
        parent = layer_that_outputs(net, var);
        if isempty(parent)
            continue;
        end
        subset = [subset, subset_to_one(net, parent)];
    end
end

function l = layer_that_outputs(net, var)
    % var is a variable name.
    l = find(arrayfun(@(l) ismember(var, l.outputs), net.layers));
    assert(numel(l) <= 1);
end
