% -------------------------------------------------------------------------------------------------------------------------
function net = remove_layers_from_prefix(net, prefix)
% -------------------------------------------------------------------------------------------------------------------------
    L = net.layers;
    num_layers = numel(L);
    to_remove = {};
    for i = 1:num_layers
        if strfind(L(i).name, prefix)
            to_remove{end+1} = L(i).name;
        end
    end
    net.removeLayer(to_remove);
end