% -------------------------------------------------------------------------------------------------------------------------
function dag_remove_except(net, keep)
% Removes all layers from net that are not in the keep set.
% keep is a list of layer indices.
% -------------------------------------------------------------------------------------------------------------------------

inds = setdiff(1:numel(net.layers), keep);
names = arrayfun(@(l) net.layers(l).name, inds, 'UniformOutput', false);
for i = 1:numel(inds)
    net.removeLayer(names{i});
end

end
