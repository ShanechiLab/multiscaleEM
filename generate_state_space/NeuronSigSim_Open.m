function [rec_spike,rec_exp_RV] = NeuronSigSim_Open(x,beta,delta)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb), Han Lin Hsieh and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is a simulation of spike neuron signal under open loop framework. Only one trial and one neuron.
% The first spike value is always zero (consider it as a determinant initial state)
% 
% Input:
% x:     the recorded (planned) kinematics along the timeline. Its dimension is (num of state)*(time length). The rows are go throught each different dimension 
%        sequentially with their related dynamics parameters (ex: dx > vx > ax > dy > vy > ay > dz > vz > ... and so on).
% beta:  the coefficients matching each states. It's a column vector with length (num of state)+1. The first coeffi. beta(1) is the baseline parameter.
% delta: the spike time interval in second.
%        
% Output:
% rec_spike:  It records the neuron spike signal along the time so its length must be the same as "time length". It is a row vector.
% rec_exp_RV: exponential R.V. record. I use -1 as the initial value for distinguishing the ones been assigned with the ones not.

% First, I set up some initial parameters.

time_length = size(x,2);
rec_spike = zeros(1,time_length); % I assume the spike value at time=0 (rec_spike(1)) is always 0.
rec_exp_RV = repmat(-1,[1,time_length]);

spike_index = 1;

% Now, I use a loop to find the next spike index continuously till it over the number of state.  

two_points_integral = [0,0]; % set up the vector for recording two boundary points in the trapezoid integral.
two_points_integral(1) = exp([1;x(:,spike_index)].'*beta); % set up the initial value for the integration.

while(spike_index < time_length)

    exp_RV = exprnd(1);
    rec_exp_RV(spike_index+1) = exp_RV;
    partial_sum = 0;
    
    while (partial_sum < exp_RV && spike_index < time_length)
        spike_index = spike_index+1;
        two_points_integral(2) = exp([1;x(:,spike_index)]'*beta); % I update the second term in the integral one step.
        partial_sum = partial_sum+trapz(two_points_integral)*delta;
        two_points_integral(1) = two_points_integral(2);
    end
    
    if (partial_sum >= exp_RV)
        rec_spike(spike_index)=1;
    end
    
end


end