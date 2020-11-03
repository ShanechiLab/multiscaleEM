function [ REALIZ,TRUE,comp] = build_statespace_realiz( OPTIONS,vhandles )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function generates state space realization from fixed settings
% INPUTS : (1)OPTIONS: a struct with following fields
%           -delta: time step
%           -scale: scale of observations, 'MS', 'Spike', 'LFP'
%           -dim_inp: arbitrary dimension of input
%           -Ttrain: number of training samples
%            -dim_Y: number of lfp features
%            -normalization_c: how to normalize C matrix for emmision
%                              matrix of lfp features
%            -scale_dif: scale difference between spiking and lfp: k
%                         spikes are available every time step and lfp are available
%                         every k steps, k, 2k, 3k, ...
%            -dim_N: number of spiking neurons
%            -spike_modulation: set to 'modesonly'
%           (2)vhandles:
%             -dim: dimension of latent state
%             -Apolehandles
%                 Apolehandles.freq: frequency of modes
%                 Apolehandles.drawing : 'fixed'
%                 Apolehandles.decay: decay of modes
%                 Apolehandles.delta : OPTIONS.delta
%             -v_R_range
%                 v_R_range.Rmean: mean value of variance (diagonal
%                 entries) of noise covariance matrix
%                 v_R_range.Rstd: mean value of variance (diagonal
%                 entries) of noise covariance matrix
%             -v_c_mod
%                 v_c_mod.mag: coeficient multiplied by C matrix
%             -v_theta_mod
%                 v_theta_mod.modulation_depth_mean: mean of modulation vector of modes
%                 for different neurons
%                 v_theta_mod.modulation_depth_std: std of modulation vector of modes
%                 for different neurons
%                 v_theta_mod.modulation_mean_mean: mean of mean firing
%                 rate for different neurons
%                 v_theta_mod.modulation_mean_std: std of mean firing
%                 rate for different neurons
%
% OUTPUTS:     (1) REALIZ: a struct contating realized values of spike and lfp, with these fields
%                     - states: latent states time series
%                     - input: input time series
%                     - N_Obs: spiking time series
%                     - FR_Obs: firing rate time series
%                     - Y_Obs: lfp features time series
%              (2) TRUE: struct contatining true model parameters
%              (3) comp: struct contatining eigenvalues and some featuers of the state space model
%% get the state space parameters range
OPTIONS.dim_hid = vhandles.dim;%dimension of system
Apolehandles = vhandles.Apolehandles;
v_R_range = vhandles.v_R_range;
v_theta_mod = vhandles.v_theta_mod;
v_c_mod = vhandles.v_c_mod;
%% set up A
phandles = Apolehandles;
phandles.drawing = 'fixed';
[EIG_A,EIG_Adis,MODES_A_othermodes] = Generate_Poles_discreteplane(phandles);% generate poles
Acont  = Acont_fromEIG( EIG_A);
Adis2 = exp( Acont * OPTIONS.delta );
[~,Adis_othermodes] = cdf2rdf(eye(OPTIONS.dim_hid),diag(EIG_Adis));
Adis = Adis_othermodes;
%% set up Q and B
Q_othermodes = diag(random('Normal',0.2,0.05,1,OPTIONS.dim_hid));
B_othermodes = 0 * randn(OPTIONS.dim_hid,OPTIONS.dim_inp);
%% set up input
Input_othermodes = zeros(OPTIONS.dim_inp,OPTIONS.Ttrain);
%% set Q B Input for total modes
Q = Q_othermodes;
B = B_othermodes;
Input = Input_othermodes;
%% generate the forward states
start_states = zeros(OPTIONS.dim_hid,1);
states_othermodes = state_generator( Adis,B,Q,Input,start_states,OPTIONS.Ttrain );
%% build states
states = states_othermodes;
%% plotting states
if 0
    figure;
    for di = 1:OPTIONS.dim_hid
        st = 1;
        fi = 32000;
        mode = floor((di-1)/2) + 1;
        %X = sin(2*pi*0.5*(st:fi)*OPTIONS.delta);
        X = states(di,st:fi);
        %figure;
        subplot(OPTIONS.dim_hid,2,2*di-1)
        plot((st:fi),X);
        title(sprintf('dimension %d--mode %d -- \n freq. = %.3g  decay = %.3g',di,mode,MODES_A_othermodes(mode).freq,MODES_A_othermodes(mode).decay));
        subplot(OPTIONS.dim_hid,2,2*di)
        X = states(di,:)-mean(states(di,:));  % states(di,:)-mean(states(di,:));
        [pxx,f] = pwelch(X,20000,[],20000,(1/OPTIONS.delta));
        %[pxx,f] = pmtm(X,4,512*64,(1/OPTIONS.delta));
        plot(f,pow2db(pxx))
        [maxpxx,index_max] = min(-pxx);
        maxpxx = -maxpxx;
        p.LineWidth = 2;
        box off;
        xlabel('freq');
        xlim([0,10])
        ylabel('Power in db');
        title(sprintf(' max in %.3g freq with %.2g db power',f(index_max),pow2db(maxpxx)));
        
    end
    
end

%% start setting up observation model
if strcmp(OPTIONS.scale,'LFP') || strcmp(OPTIONS.scale,'MS')
    %% set up parameters for observations matrix
    %1. set up C and R and bias and D
    C_temp = random('Uniform',0.5,6,OPTIONS.dim_Y,OPTIONS.dim_hid); %linear observation matrix
    %C_temp = random('Normal',0,3,OPTIONS.dim_Y,OPTIONS.dim_hid); %linear observation matrix
    %OPTION#1
    if strcmp(OPTIONS.normalization_c,'std')
        std_states = sqrt(diag(cov(states')));
    elseif strcmp(OPTIONS.normalization_c,'peaktopeak')
        %OPTIONS#2
        std_states = sqrt (  ( max(states,[],2)-min(states,[],2) )/2 ) ;
    end
    %contribution_states_in_C = 1 * ones(1,OPTIONS.dim_hid);
    contribution_states_in_C = 1./std_states';
    contribution_states_in_C = v_c_mod.mag * contribution_states_in_C; % to level up the observations dominance
    C = C_temp .* repmat(contribution_states_in_C,OPTIONS.dim_Y,1);
    
    %% R and bias and D
    R = diag(random('Normal',v_R_range.Rmean,v_R_range.Rstd,OPTIONS.dim_Y,1));
    bias = zeros(OPTIONS.dim_Y,1);
    D = zeros(OPTIONS.dim_Y,OPTIONS.dim_inp);
    %% generate linear observations from parameters
    Y_Obs = linear_obs_generator( states,C,D,R,Input,bias );
    
end
%% generate spike modulation matrix and spiking activity
if strcmp(OPTIONS.scale,'LFP')
    
    N_Obs = NaN(1,OPTIONS.Ttrain);
    FR_Obs = NaN(1,OPTIONS.Ttrain);
    
elseif strcmp(OPTIONS.scale,'MS') || strcmp(OPTIONS.scale,'Spike')
    
    if strcmp(OPTIONS.spike_modulation,'modesonly')
        handles_theta = struct;
        handles_theta.modulation_depth_mean = v_theta_mod.modulation_depth_mean;
        handles_theta.modulation_depth_std = v_theta_mod.modulation_depth_std;
        
        handles_theta.modulation_mean_mean = v_theta_mod.modulation_mean_mean;
        handles_theta.modulation_mean_std = v_theta_mod.modulation_mean_std;
        
        
        [ Theta_kin,Theta_modes ] = theta_generator_onlymodes( states,OPTIONS.dim_N,OPTIONS.spike_modulation,handles_theta );
        Theta = zeros(OPTIONS.dim_hid+1,OPTIONS.dim_N);
        Theta(1,:) = Theta_kin(1,:);
        Theta(2:end,:) =  Theta_modes;
    end
    
    
    [N_Obs,FR_Obs] = spike_generator(Theta,states,OPTIONS.delta);
    max_FR_pern = NaN(OPTIONS.dim_N,1);
    N_Obs = squeeze(N_Obs);
    % plot max firing rate
    for j = 1:OPTIONS.dim_N
        max_FR_pern(j) = max(FR_Obs(j,:));
        
    end
    if 0
        figure;
        plot(1:OPTIONS.dim_N, max_FR_pern)
        title('max_FR')
    end
    
end

if strcmp(OPTIONS.scale,'Spike')
    
    Y_Obs = NaN(1,OPTIONS.Ttrain);
    
end

%% sets up  TRUE struct
TRUE = struct;
TRUE.A = Adis;
TRUE.B = B;
TRUE.Q = Q;
if strcmp(OPTIONS.scale,'MS') || strcmp(OPTIONS.scale,'LFP')
    TRUE.C = C;
    TRUE.D = D;
    TRUE.R = R;
    TRUE.bias = bias;
end
if strcmp(OPTIONS.scale,'MS') || strcmp(OPTIONS.scale,'Spike')
    TRUE.Theta = Theta;
end
TRUE.vhandles = vhandles;
%% sets up comp struct

comp = struct;
%comp.MODES_A = MODES_A;
comp.EIG_A = eig(Adis);
comp.EIG_Adis = eig(Adis);
comp.MODES_A_othermodes  = MODES_A_othermodes;
comp.info = struct;
comp.info.Apolehandles = Apolehandles;
if strcmp(OPTIONS.scale,'MS') || strcmp(OPTIONS.scale,'Spike')
    comp.FR = struct;
    comp.FR.FR_Obs = FR_Obs;
end
%% sets up REALIZ
REALIZ = struct;
REALIZ.states = states;
REALIZ.input = Input;
REALIZ.Y_Obs = Y_Obs;
REALIZ.N_Obs = N_Obs;
REALIZ.FR_Obs = FR_Obs;


end

