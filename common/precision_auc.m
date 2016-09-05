function score = precision_auc(positions, groundTruth, radius, nStep)
%PRECISION_PLOT
%   Calculates precision for a series of distance thresholds (percentage of
%   frames where the distance to the ground truth is within the threshold).
%   The results are shown in a new figure if SHOW is true.
%
%   Accepts positions and ground truth as Nx2 matrices (for N frames)
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	thresholds = linspace(0,radius,nStep);
    %TODO: what about choosing steps more coarsely while going further from the center?
	errs = zeros(nStep, 1);

	%calculate distances to ground truth over all frames
	distances = sqrt((positions(:,1) - groundTruth(:,1)).^2 + ...
				 	 (positions(:,2) - groundTruth(:,2)).^2);
	distances(isnan(distances)) = [];

	%compute precisions
	for p = 1:nStep
		errs(p) = nnz(distances > thresholds(p));
	end

    % Area Under the Curve (AUC) approximation, normalized by best (approx) score
	score = trapz(errs);
end