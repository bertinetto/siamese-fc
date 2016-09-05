% -------------------------------------------------------------------------------------------------------
function [cx, cy, w, h] = get_rect(object)
%GET_RECT
%	Converts from original frame coordinates to resized one.
% -------------------------------------------------------------------------------------------------------
    xmin = round(object.extent(1) * object.new_frame_sz(1) / object.frame_sz(1));
    ymin = round(object.extent(2) * object.new_frame_sz(2) / object.frame_sz(2));
    w = round(object.extent(3) * object.new_frame_sz(1) / object.frame_sz(1));
    h = round(object.extent(4) * object.new_frame_sz(2) / object.frame_sz(2));
    cx = xmin + w/2;
    cy = ymin + h/2;
end