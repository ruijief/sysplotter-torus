function output = sysf_three_link_lowRe_CoM(input_mode,pathnames)

	% Default arguments
	if ~exist('input_mode','var')
		
		input_mode = 'initialize';
		
	end
		
	%%%%%%%
	
	switch input_mode

		case 'name'

			output = 'Viscous Swimmer: 3-link CoM'; % Display name

		case 'dependency'

			output.dependency = fullfile(pathnames.sysplotterpath,...
                {'Utilities/curvature_mode_toolbox/N_link_chain.m';...
				'Utilities/LowRE_toolbox/LowRE_dissipation_metric_discrete.m';...
				'Utilities/LowRE_toolbox/LowRE_local_connection_discrete.m'});

		case 'initialize'

            %%%%%%
            % Define system geometry
            s.geometry.type = 'n-link chain';
            s.geometry.linklengths = [1 1 1];
            s.geometry.baseframe = 'com-mean';
            s.geometry.length = 1;
            
            
            %%%%%%
            % Define system physics
            s.physics.drag_ratio = 2;
            s.physics.drag_coefficient = 1;
           
 
            %Functional Local connection and dissipation metric

            s.A = @(alpha1,alpha2) LowRE_local_connection_discrete( ...
                        s.geometry,...                           % Geometry of body
                        s.physics,...                            % Physics properties
                        [alpha1,alpha2]);                        % Joint angles
            
            s.metric = @(alpha1,alpha2) LowRE_dissipation_metric_discrete(...
                        s.geometry,...                           % Geometry of body
                        s.physics,...                            % Physics properties
                        [alpha1,alpha2]);                        % Joint angles

                    
			%%%
			%Processing details

			%Range over which to evaluate connection
			s.grid_range = [-1,1,-1,1]*2.5;

			%densities for various operations
			s.density.vector = [11 11 ]; %density to display vector field
			s.density.scalar = [21 21 ]; %density to display scalar functions
			s.density.eval = [51 51 ];   %density for function evaluations
            s.density.metric_eval = [11 11]; %density for metric evaluation
			s.finite_element_density = 21;

            %%%
			%Display parameters

			%shape space tic locations
			s.tic_locs.x = [-1 0 1]*1;
			s.tic_locs.y = [-1 0 1]*1;


			%%%%
			%Save the system properties
			output = s;


	end

end

