% -------------------------------------------------------------------------------------------------------------------------
classdef XCorr < dagnn.Layer
%XCORR
%   Crosscorrelates two activations of different size exploiting  the API of vl_nnconv
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    properties
        opts = {'cuDNN'}
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 2, 'two inputs are needed');

            z = inputs{1}; % exemplar
            x = inputs{2}; % instance (search region)

            assert(ndims(z) == ndims(x), 'z and x have different number of dimensions');
            assert(size(z,1) <= size(x,1), 'exemplar z has to be smaller than instance x');

            [wx,hx,cx,bx] = size(x);
            x = reshape(x, [wx,hx,cx*bx,1]);
            o = vl_nnconv(x, z, []);
            [wo,ho,co,bo] = size(o);
            assert(co==bx);
            outputs{1} = reshape(o, [wo,ho,bo,co]);
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            assert(numel(inputs) == 2, 'two inputs are needed');
            assert(numel(derOutputs) == 1, 'only one gradient should be flowing in this layer (dldy)');
            z = inputs{1}; % exemplar
            x = inputs{2}; % instance
            assert(size(z,1) < size(x,1), 'exemplar z has to be smaller than instance x');
            [wx,hx,cx,bx] = size(x);
            x = reshape(x, [wx,hx,cx*bx,1]);
            dldy = derOutputs{1};
            [wdl,hdl,cdl,bdl] = size(dldy);
            assert(cdl==1);
            dldy = reshape(dldy, [wdl,hdl,cdl*bdl,1]);
            [dldx, dldz, ~] = vl_nnconv(x, z, [], dldy);
            [mx,nx,cb,one] = size(dldx);
            assert(mx == size(x, 1));
            assert(nx == size(x, 2));
            assert(cb == cx * bx);
            assert(one == 1);
            derInputs{1} = dldz;
            derInputs{2} = reshape(dldx, [mx,nx,cx,bx]);
            derParams = {};
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            z_sz = inputSizes{1};
            x_sz = inputSizes{2};
            y_sz = [x_sz(1:2) - z_sz(1:2) + 1, 1, z_sz(4)];
            outputSizes = {y_sz};
        end

        function rfs = getReceptiveFields(obj)
            rfs(1,1).size = [inf inf]; % could be anything
            rfs(1,1).stride = [1 1];
            rfs(1,1).offset = 1;
            rfs(2,1).size = [inf inf];
            rfs(2,1).stride = [1 1];
            rfs(2,1).offset = 1;
        end

        function obj = XCorr(varargin)
            obj.load(varargin);
        end

    end

end
