classdef centerThrErr < dagnn.Loss
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    methods
        function outputs = forward(obj, inputs, params)
            radiusInPixel = 50;
            totalStride = 8;
            nStep = 100;
            batch_size = size(inputs{1},4);
            pos_mask = inputs{2}(:) > 0;
            num_pos = sum(pos_mask);
            outputs{1} = 0;
            responses = inputs{1};
            responses = responses(:,:,:,pos_mask);
            n = obj.numAveraged;
            % avg only on num pos, not entire batch
            m = n + num_pos;
            half = floor(size(inputs{1},1)/2)+1;
            centerLabel = repmat([half half], [num_pos 1]);
            positions = zeros(num_pos, 2);
            for b = 1:num_pos
                score = gather(responses(:,:,1,b));
                [r_max, c_max] = find(score == max(score(:)), 1);
                positions(b, :) = [r_max c_max];
            end
            outputs{1} = precision_auc(positions, centerLabel, radiusInPixel/totalStride, nStep);
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end

        function obj = centerThrErr(varargin)
          obj.load(varargin) ;
        end
  end
end
