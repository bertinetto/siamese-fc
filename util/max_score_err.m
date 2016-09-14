% -------------------------------------------------------------------------------------------------------------------------
function y = max_score_err(x, y_gt)
% x is [m1, m2, 1, b]
% numel(y_gt) is b
% The dimensions m1 and m2 are odd numbers.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------

[m1, m2, k, b] = size(x);
assert(mod(m1, 2) == 1);
assert(mod(m2, 2) == 1);
assert(k == 1);

assert(numel(y_gt) == b);

y_gt = reshape(y_gt, [1, 1, 1, b]);
pos = y_gt > 0;
neg = y_gt < 0;

x = gather(x);

% % Express error e as a linear function of scores x.
% h_center = center_mask(m1, m2, b);
% h_max = max_mask(x);
% h = bsxfun(@times, pos, h_center) - bsxfun(@times, neg, h_max);
% % e is positive if classified correctly, negative if incorrectly.
% e = sum(sum(h .* x, 1), 2);

x_center = center_score(x);
x_max = max_score(x);
e = zeros(b, 1);
e(pos) = x_center(pos);
e(neg) = -x_max(neg);

y = sum(e <= 0);

end

function v = center_score(x)
    [m1, m2, ~, b] = size(x);
    c1 = (m1+1) / 2;
    c2 = (m2+1) / 2;
    v = x(c1, c2, :, :);
end

function h = center_mask(m1, m2, b)
% This should satisfy x .* center_mask(m1, m2, b) == center_score(x).
    c1 = (m1+1) / 2;
    c2 = (m2+1) / 2;
    h = zeros(m1, m2, 1, b, 'single');
    h(c1, c2, :, :) = 1;
end

function v = max_score(x)
    [m1, m2, ~, b] = size(x);
    v = max(max(x, [], 1), [], 2);
end

function h = max_mask(x)
% This should satisfy x .* max_mask(x) == max_score(x).
    [m1, m2, ~, b] = size(x);
    x = reshape(x, [m1*m2, b]);
    [~, u] = max(x);
    assert(numel(u) == b);
    h = zeros(m1*m2, b, 'single');
    h(sub2ind(size(x), u, 1:8)) = 1;
    h = reshape(h, [m1, m2, 1, b]);
end
