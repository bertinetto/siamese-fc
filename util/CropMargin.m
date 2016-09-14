% -------------------------------------------------------------------------------------------------------------------------
classdef CropMargin < dagnn.Layer
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------

  properties
    margin
  end

  methods
    function outputs = forward(obj, inputs, params)
      assert(numel(inputs) == 1);
      assert(numel(params) == 0);
      x = inputs{1};
      sz = size_min_ndims(x, 4);
      p = obj.margin;
      y = x(1+p:end-p, 1+p:end-p, :, :);
      outputs = {y};
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      assert(numel(inputs) == 1);
      assert(numel(params) == 0);
      assert(numel(derOutputs) == 1);
      x = inputs{1};
      dldy = derOutputs{1};
      if isa(x, 'gpuArray')
        dldx = gpuArray(zeros(size(x), classUnderlying(x)));
      else
        dldx = zeros(size(x), class(x));
      end
      p = obj.margin;
      dldx(1+p:end-p, 1+p:end-p, :, :) = dldy;
      derInputs = {dldx};
      derParams = {};
    end

    % function kernelSize = getKernelSize(obj)
    %   kernelSize = obj.size(1:2);
    % end

    function outputSizes = getOutputSizes(obj, inputSizes)
      p = obj.margin;
      x_sz = inputSizes{1};
      y_sz = max(0, x_sz - [2*p, 2*p, 0, 0]);
      outputSizes = {y_sz};
    end

    function rfs = getReceptiveFields(obj)
      rfs(1,1).size = [1 1];
      rfs(1,1).stride = [1 1];
      rfs(1,1).offset = 1 + obj.margin;
    end

    function obj = CropMargin(varargin)
      obj.load(varargin);
    end
  end
end
