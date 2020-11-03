function [ N_Obs,FR ] = spike_generator(Theta,states,delta)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function generates  N_Obs and Firing Rate from states
% INPUTS:     -Theta: spike modulation matrix, (dim + 1, N), where dim is the dimension
%                     of the latent state, N is the number of neurons
%             - states: latent states, (dim, T)
%             - delta: time step
% 
% OUTPUT:     - N_Obs: discrete spikes, (N, T)
%             - FR: firing rate of spikes, (N, T)

%% get some values
N=size(Theta,2);
Tmain = size(states,2);
[dim,~]=size(states);
FR=NaN(N,Tmain);
%% generate firing rate
for i=1:Tmain    
    for j=1:N
        FR(j,i)=exp(Theta(1,j)' + Theta(2:dim+1,j)' * states(:,i)) * delta;
    end
end
%% generate spikes
%use this function to generate spikes
[N_Obs,~]=NeuronSigSim_Multi_Open(states,Theta,delta,1);


end




