function result = isToolboxAvailable(toolboxName,action)
%% check if an specific toolbox exists
%
% This small piece of code checks to find whether a specific toolbox is
% installed on MATLAB.
%
% The input is the name of the toolbox as a string
%
% The output is 1 if the toolbox is found and 0 otherwise
%
% Example:
%
%   result = isToolboxAvailable('Image Processing Toolbox','warning')
%
% -----------------------------------------------
% code by: Reza Ahmadzadeh (reza.ahmadzadeh@iit.it)
% -----------------------------------------------
%
%% check the input arguments
if nargin < 1 || nargin > 2
    error('usage: isToolboxAvailable(toolboxName,action)');
elseif nargin == 1
    action = 'warning';
end;

%% Find the toolbox and give proper output
v_=ver;
[installedToolboxes{1:length(v_)}] = deal(v_.Name);
result = all(ismember(toolboxName,installedToolboxes));
switch action
    case 'error'
        % if you need to end the program you can use the following line of code
        assert(result,['Error! ' toolboxName ' is not installed!']);
    case 'warning'
        % if you need to just give a warning you can use the following line of code
        if ~result
            warning([toolboxName ' is not installed!']);
        end
    otherwise
end
end