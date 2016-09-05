% -------------------------------------------------------------------------------------------------------------------------
function y = gaussian_weight(rect_size, sigma)
% -------------------------------------------------------------------------------------------------------------------------
    % a false positive should be penalized more if far from the center
	y = zeros(rect_size(1),rect_size(2));
    half = floor((rect_size+1) / 2);
    i_range = (1:rect_size(1)) - half(1);
    j_range = (1:rect_size(2)) - half(2);
    for i=1:rect_size(1)
        for j=1:rect_size(2)
            y(i, j) = 1 - exp(-(i_range(i).^2 + j_range(j).^2) / (2 * sigma^2));
        end
    end
	assert(sum(y(half(1), half(2)))==0);
    y(half(1), half(2)) = 1;
end

