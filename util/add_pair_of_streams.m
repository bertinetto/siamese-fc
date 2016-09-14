% -------------------------------------------------------------------------------------------------------------------------
function net = add_pair_of_streams(net, stream, inputs, outputs, siamese)
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    name_a = @(s) ['a_', s];
    name_b = @(s) ['b_', s];
    rename_a = struct('layer', name_a, 'var', name_a, 'param', name_a);
    rename_b = struct('layer', name_b, 'var', name_b, 'param', name_b);
    share = @(s) s;
    if siamese
        rename_a.param = share;
        rename_b.param = share;
    end

    add_dag_to_dag(net, stream, rename_a);
    add_dag_to_dag(net, stream, rename_b);
    net.renameVar(name_a('in'), inputs{1});
    net.renameVar(name_b('in'), inputs{2});
    net.renameVar(name_a('out'), outputs{1});
    net.renameVar(name_b('out'), outputs{2});
end