% -------------------------------------------------------------------------------------------------------------------------
function logloss_label = create_logisticloss_label(label_size, rPos, rNeg)
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    % contruct label for logistic loss (same for all pairs)
    label_side = label_size(1);
    logloss_label = single(zeros(label_side));
    label_origin = [ceil(label_side/2) ceil(label_side/2)];
    for i=1:label_side
        for j=1:label_side
            dist_from_origin = dist([i j], label_origin');
            if dist_from_origin <= rPos
                logloss_label(i,j) = +1;
            else
                if dist_from_origin <= rNeg
                    logloss_label(i,j) = 0;
                else
                    logloss_label(i,j) = -1;
                end
            end
        end
    end
end
