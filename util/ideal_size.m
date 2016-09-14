% -------------------------------------------------------------------------------------------------------------------------
function [init_sz, final_sz] = ideal_size(net, max_sz)
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
final_sz = forward(net, max_sz);
init_sz = backward(net, final_sz);
while ~all(init_sz <= max_sz)
    final_sz = final_sz - 1;
    init_sz = backward(net, final_sz);
end

end

function n = forward(net, n)
    for i = 1:numel(net.layers)
        l = net.layers{i};
        switch l.type
        case 'conv'
            m = [size(l.weights{1}, 1), size(l.weights{1}, 2)];
            n = filter(n, l.pad, m, l.stride);
        case 'pool'
            n = filter(n, l.pad, l.pool, l.stride);
        end
    end
end

function n = backward(net, n)
    for i = numel(net.layers):-1:1
        l = net.layers{i};
        switch l.type
        case 'conv'
            m = [size(l.weights{1}, 1), size(l.weights{1}, 2)];
            n = unfilter(n, l.pad, m, l.stride);
        case 'pool'
            n = unfilter(n, l.pad, l.pool, l.stride);
        end
    end
end

function n = filter(n, pad, m, k)
assert(numel(pad) == 1);
n = floor((n + 2*pad - m) / k) + 1;
end

function n = unfilter(n, pad, m, k)
assert(numel(pad) == 1);
n = k*(n - 1) + m - 2*pad;
end
