% -------------------------------------------------------------------------------------------------------
function [fixedLabel, instanceWeight] = create_labels(fixedLabelSize, labelWeight, rPos, rNeg)
%CREATE_LABELS
%
%   Luca Bertinetto, Jack Valmadre, Joao Henriques, 2016
% -------------------------------------------------------------------------------------------------------
    assert(mod(fixedLabelSize(1),2)==1)
    half = floor(fixedLabelSize(1)/2)+1;
    switch labelWeight
        case 'uniform'
            % constant weight
            fixedLabel = create_logisticloss_label(fixedLabelSize, rPos, rNeg);
            instanceWeight = 1/nnz(fixedLabel) * ones(size(fixedLabel));
        case 'balanced' % default choice
            % weight by class cardinality (+/-)
            fixedLabel = create_logisticloss_label(fixedLabelSize, rPos, rNeg);
            instanceWeight = ones(size(fixedLabel));
            sumP = numel(find(fixedLabel==1));
            sumN = numel(find(fixedLabel==-1));
            instanceWeight(fixedLabel==1) = 0.5 * instanceWeight(fixedLabel==1) / sumP;
            instanceWeight(fixedLabel==-1) = 0.5 * instanceWeight(fixedLabel==-1) / sumN;
        case 'gaussian'
            % gaussian weights
            fixedLabel = -1 * ones(fixedLabelSize(1));
            fixedLabel(half, half) = 1;
            instanceWeight = gaussian_weight(fixedLabelSize', fixedLabelSize(1)/2);
            instanceWeight = instanceWeight / sum(instanceWeight(:));
        case 'neutral_area'
            % separate positive (+1) and negatives (-1) by an area of neutral (0)
            fixedLabel = -1 * ones(fixedLabelSize(1));
            fixedLabel(half, half) = 1;
            instanceWeight = create_logisticloss_label(fixedLabelSize, 0, 2);
            instanceWeight(instanceWeight==-1) = 1;
            sumPos = sum(fixedLabel(instanceWeight==1)==+1);
            sumNeg = sum(fixedLabel(instanceWeight==1)==-1);
            instanceWeight(fixedLabel==+1) = 0.5 * instanceWeight(fixedLabel==1) / sumPos;
            instanceWeight(fixedLabel==-1) = 0.5 * instanceWeight(fixedLabel==-1) / sumNeg;
        otherwise
            error('Unknown option for instance weights');
    end
end
