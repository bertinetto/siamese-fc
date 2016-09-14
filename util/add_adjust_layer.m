% -------------------------------------------------------------------------------------------------------------------------
function net = add_adjust_layer(net, name, input, output, params, gain, bias, lr_gain, lr_bias)
% -------------------------------------------------------------------------------------------------------------------------
    net.addLayer(name, ...
                 dagnn.Conv('size', [1, 1, 1, 1]), ...
                 {input}, {output}, ...
                 params);
    filters = net.getParamIndex(params{1});
    biases = net.getParamIndex(params{2});
    net.params(filters).value = single(gain);
    net.params(biases).value = single(bias);
    net.params(filters).learningRate = lr_gain;
    net.params(biases).learningRate = lr_bias;
end