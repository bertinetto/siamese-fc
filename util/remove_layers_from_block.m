% -------------------------------------------------------------------------------------------------------------------------
function net = remove_layers_from_block(net, type)
% -------------------------------------------------------------------------------------------------------------------------
  [names, ~] = find_layers_from_block(net, type);
  for i = 1:numel(names)
    layer = net.layers(net.getLayerIndex(names{i}));
    net.removeLayer(names{i});
    net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true);
  end
end
