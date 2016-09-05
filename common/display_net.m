% -------------------------------------------------------------------------------------------------------------------------
function display_net(net, inputs, name)
%DISPLAY_NET
%   Saves and display a pdf with the DAG structure of the network
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    if strcmp(class(net), 'struct')
        net = dagnn.DagNN.fromSimpleNN(net);
    end
    obj = net.saveobj();
    name_mat = sprintf('%s.mat', name);
    name_dot = sprintf('%s.dot', name);
    name_ps = sprintf('%s.ps', name);
    save(name_mat, '-struct', 'obj');
    model2dot(name_mat, name_dot, 'inputs', inputs);
    dot2ps = sprintf('dot -Tps %s -o %s', name_dot, name_ps);
    unix(dot2ps);
    showpdf = sprintf('evince %s &', name_ps);
    unix(showpdf);
end