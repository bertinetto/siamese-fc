% -------------------------------------------------------------------------------------------------------------------------
function [layer_names, layer_ids] = find_layers_from_block(net, type)
% -------------------------------------------------------------------------------------------------------------------------
% works with simplenn
  layer_names = [];
  layer_ids = [];
  for l = 1:numel(net.layers)
    if isa(net.layers(l).block, type)
      layer_names{end+1} = net.layers(l).name;
	  layer_ids(end+1) = l;
    end
  end
end