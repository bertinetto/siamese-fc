% -------------------------------------------------------------------------------------------------------------------------
classdef correctIncorrect < dagnn.Loss
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    methods
        function outputs = forward(obj, inputs, params)
            batch_size = size(inputs{1},4);
            pos_neg_ids = zeros(1, batch_size);
            %TODO: pass opts.negatives from experiment.m
            % check which pairs are negative
            for b=1:batch_size
                if(numel(find(inputs{2}(:,:,1,b)==1))>0)
                    pos_neg_ids(b) = 1;
                end
            end
            num_pos = sum(pos_neg_ids);
            num_neg = batch_size - num_pos;
            n = obj.numAveraged;
            % avg only on num pos, not entire batch
            m = n + num_pos;
            negs = find(pos_neg_ids==0);
            peaks = zeros(1, batch_size);
            % count how many pos pairs have a peak lower than a rnd neg pair
            for b = 1:batch_size
                response = gather(inputs{1}(:,:,1,b));
                peaks(b) = max(response(:));
            end
            outputs{1} = 0;
            for p = 1:num_pos
                outputs{1} = outputs{1} + numel(find(peaks(negs)>=peaks(p)))/num_neg*100;
            end
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end

        function obj = centerErr(varargin)
          obj.load(varargin) ;
        end
  end
end