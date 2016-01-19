classdef fighandle < handle
    % Figure handle class for imshow3f, imshow_flow and imshow_flowmag
    % Its only use is to store parameters
    properties
        figure      % figure
        fh          % figure handle for image
        a           % axis handle
        x           % xSlice
        z           % slice
        c           % cslice
        map         % contrast window
        singleSlice % is single slice?
        subsample   % subsample flow field
        vmap        % velocity window
        scale_vec   % scale vector
        imThresh    % image threshold for flow
        xgrid       % x grid for flow vectors
        ygrid       % y grid for flow vectors
        fh_flow     % figure handle for flow vectors
        busy
    end
    
    methods
        function obj = fighandle ()

        end
    end
end