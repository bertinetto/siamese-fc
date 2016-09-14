% From examples/imagenet/cnn_imagenet_deploy.m
% -------------------------------------------------------------------------
function dagRemoveLayers(net, names)
% -------------------------------------------------------------------------
for i = 1:numel(names)
  layer = net.layers(net.getLayerIndex(names{i})) ;
  net.removeLayer(names{i}) ;
  net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end
