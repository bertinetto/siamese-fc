% -------------------------------------------------------------------------------------------------------------------------
classdef MaxScoreErr < dagnn.Loss
%MAXSCOREERR computes binary error of a response map.
% It uses the maximum response of a negative example
% and the response at the center for a positive example.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
  methods
    function outputs = forward(obj, inputs, params)
    % inputs{1} is a scalar response map
    % inputs{2} is a scalar label, -1 or 1
      outputs{1} = max_score_err(inputs{1}, inputs{2});
      n = obj.numAveraged;
      m = n + size(inputs{1},4);
      obj.average = (n * obj.average + gather(outputs{1})) / m;
      obj.numAveraged = m;
    end

    function reset(obj)
      obj.average = 0;
      obj.numAveraged = 0;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)];
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN];
      rfs(1,1).stride = [NaN NaN];
      rfs(1,1).offset = [NaN NaN];
      rfs(2,1) = rfs(1,1);
    end

    function obj = MaxScoreErr(varargin)
      obj.load(varargin);
    end
  end
end
