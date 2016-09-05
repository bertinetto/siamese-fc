function run_experiment(imdb_video)
%RUN_EXPERIMENT
%	contains the parameters that are specific to the current experiment.
% 	These are the parameters that change frequently and should not be committed to
% 	the repository but should be saved with the results of the experiment.

	% Parameters that should have no effect on the result.
	opts.prefetch = false;
	opts.gpus = 1;

	% Parameters that should be recorded.
	% opts.foo = 'bla';

	if nargin < 1
	    imdb_video = [];
	end
	experiment(imdb_video, opts);

end

