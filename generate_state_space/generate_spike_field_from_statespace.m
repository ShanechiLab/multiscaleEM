function [ REALIZ, TRUE ] = generate_spike_field_from_statespace( OPTIONS )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function generates the spike lfp activtiy from a simulated state space
% INPUTS:     - OPTIONS: a struct contatining options of the state space
%                 -delta: time step
%                 -scale: scale of observations, 'MS', 'Spike', 'LFP'
%                 -dim_inp: arbitrary dimension of input
%                 -Ttrain: number of training samples
%                 -dim_Y: number of lfp features
%                 -normalization_c: how to normalize C matrix for emmision
%                                  matrix of lfp features
%                 -scale_dif: scale difference between spiking and lfp: k
%                             spikes are available every time step and lfp are available
%                             every k steps, k, 2k, 3k, ...
%                 -dim_N: number of spiking neurons
%                 -spike_modulation: set to 'modesonly' 
% 
% OUTPUTS:    - REALIZ: a struct contating realized values of spike and lfp, with these fields
%                 - states: latent states time series
%                 - input: input time series
%                 - N_Obs: spiking time series
%                 - FR_Obs: firing rate time series
%                 - Y_Obs: lfp features time series
%             - TRUE: struct contatining true model parameters
%             - comp: struct contatining eigenvalues and some featuers of the state space model
%% generate the eigenvalues (change this if you want to change eigenvalues)
% generate eigenvalues (way 1)
% dim_can = 8;
% [eigVals,res] = selectRandomPoles([30*3 30*2 30], [0.995 0.96 0.90], pi/500 * ones(3,1), [], dim_can);
% res.poles(res.sysPoleInd, :); %poles
% decay_can = ((OPTIONS.delta)) ./ ( log(1./res.allMags(res.sysPoleInd, :)) );
% freq_can =  res.allTheta(res.sysPoleInd, :)/ (2 * pi * OPTIONS.delta);
% generate eigenvalues (way 2, fixed)
dim_can = 8;
decay_can = [0.6,0.07,0.1,0.8];
freq_can =  [0.3,2.8,1,2];
Apolehandles = struct;Apolehandles.freq = freq_can;
Apolehandles.decay = decay_can;Apolehandles.drawing = 'fixed';Apolehandles.delta = OPTIONS.delta;
%% generate range of R, Q, C and Theta (read build_statespace_realiz.m for more info)
R_range_can = struct;R_range_can.Rmean = 1500;R_range_can.Rstd = 100;
Theta_mod_can = struct;Theta_mod_can.modulation_depth_mean = 1.8;
Theta_mod_can.modulation_depth_std = 0.1; Theta_mod_can.modulation_mean_mean = 2;
Theta_mod_can.modulation_mean_std = 0.1;
C_mod_can = struct;C_mod_can.mag = 3;
vhandles = struct;
vhandles.dim = dim_can;
vhandles.Apolehandles = Apolehandles;
vhandles.v_R_range = R_range_can;
vhandles.v_c_mod = C_mod_can;
vhandles.v_theta_mod = Theta_mod_can;    
%% generate system
[ REALIZ,TRUE,~] = build_statespace_realiz( OPTIONS,vhandles );

end

