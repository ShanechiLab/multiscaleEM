function [rec_spike,rec_exp_RV] = NeuronSigSim_Multi_Open(x,beta,delta,trials)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb), Han Lin Hsieh and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is a simulation of spike neuron signal under open loop framework. It can do multiple times simulation with multiple neurons all at once.
% The first spike value is always zero (consider it as a determinant initial state)
% 
% Input:
% x:      the recorded (planned) kinematics along the timeline. Its dimension is (num of state)*(time length). The rows are go throught each different dimension 
%         sequentially with their related dynamics parameters (ex: dx > vx > ax > dy > vy > ay > dz > vz > ... and so on).
% beta:   the coefficients matching each states. Its dimension is (num of state+1)*(neuron num). Every column represents a coefficient set of one neuron. The
%         first coeffi. row beta(1,:) is the baseline parameter.
% delta:  the spike time interval in second.
% trials: the number of trials under the same planning dynamics and beta coefficients.
%        
% Output:
% rec_spike:  It records the neuron spike signal of each neuron along the time in each trial. Its dimension is (trials)*(neuron num)*(time length).
% rec_exp_RV: exponential R.V. record. Its dimension is (trials)*(neuron num)*(time length).

time_length = size(x,2);
neuron_num = size(beta,2);

rec_spike = zeros(trials,neuron_num,time_length); % I assume the spike value at time=0 (rec_spike(1)) is always 0.
rec_exp_RV = zeros(trials,neuron_num,time_length);

for(i=1:1:trials)
    for(j=1:1:neuron_num)
        [rec_spike(i,j,:),rec_exp_RV(i,j,:)]=NeuronSigSim_Open(x,beta(:,j),delta);
    end
end


end