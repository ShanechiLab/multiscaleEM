function [ results,settings,ITER ] = EM_multiscale_unsupervised_function( Y_Obs,N_Obs,handles )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function learns the unsupervised latent state space model from multiscale spiking and
% LFP activity. The details of the derivation could be found in this paper:
% 10.1109/TNSRE.2019.2913218
%
% In general, the state-space model is learned through an iterative expectation-maximization algorithm
% the state space model is as follows:
%
% x_{t + 1} = A * x_{t} + q_t; COV(q_t) = Q
% y_{t} = C * x_{t} + r_t; COV(r_t) = R
% p(N_{t}|x_{t}) = (\lambda(x_{t})) ^ (N_{t}) * exp( -\lambda(x_{t}) * \Delta ); \lambda(x_{t}) = exp(-\alpha*x_{t} + \beta), \Delta: time-step
%
% An example sample use of this algorithm can be found in ''.
%
% INPUTS :
%         (1) Y_Obs: time series of the LFPs, its dimension is (dim_Y*T), where dim_Y is the number of channles and T is
%         the total number of samples! if time-scale of LFP is k (available
%         at k,2k,3k,...), it is recommended to put the samples in between as NaN, however the model will never use them
%         (2) N_Obs: time series of spikes with size N*T, N is number of neurons
%         (3) handles: consists some options for the EM algorithm and for additional options in the future (struct with following fields)
%
%             -scale_dif_inp : scale difference in spikes and LFPs, k, i.e., spikes are available at every time-step and lfp are available
%                             only at k, 2k, 3k, ...
%             -delta_inp :time step of the spike bins (or real time-scale of sampling/dynamics) (in seconds)
%             -dim_hid: dimension of hidden state
%             -num_iter: number of iterations to run the EM algorithm for
%             -switch_nondiag: if it is 1 it means the observation noise will be learned as a non-diagonal covariance matrix, if it is 0 the observation noise is diagonal
%             -switch_nondiagQ: (Always put this as 1, for fast learning!) if it is 1 it means the state noise is non-diagonal, if it is 0 the state noise is diagonal
%             -init_type: what type of initialization to use ('random' or 'subid'), if it is 'random' the model randomly initialize the parameters at iter 1, if not
%                         if not it uses subspace identification to
%                         initialize parameters, for now always set to 'random'
%             -switch_biasobs: whether to fit bias for linear observations, if 1, it learns a linear bias for Y_Obs
%             -save_iter: which iterations to save at ITER output: 'all' or 'startandend'. 'all' saves all iterations, 'startandend' saves only the first and last
%                         iteration
%             -spike_bs_init: initialization for spike baseline: 'random' or 'meanFR', whehter to initialize \beta for spikes with random value or mean firing rate of
%                             observations
% OUTPUTS:
%         (1)results : final results of state space with smoothed and filtered states using that state space, an struct with the following fields
%                     - A: state transition matrix
%                     - B: state-input matrix (default: zeros/not learned)
%                     - Q: state noise covariance matrix
%                     - C: observation emission matrix
%                     - D: observation-input matrix (default: zeros/not learned)
%                     - R: observation noise covariance matrix
%                     - Init_X: initial estimation of the latent state at t = 1
%                     - Init_Cov: initial estimation of the latent state estimation error at t = 1
%                     - Theta: parameters of spike modulation -> [\beta_c;alpha_c] in each column for every neuron
%                     - Bias: bias of lfp observation
%                     - X_update: filtered (causal inference) values of latent state in training with the final parameters
%                     - X_smoothed: smoothed (non-causal inference) values of latent state in training with the final parameters
%         (2)settings: settings used to run the EM algorithm (similar to handles)
%         (3)ITER: saves the parameters of intermediate steps as well with the following fields
%                     -iter: struct array with following fields at each index
%                         - similar to results: A, B, Q, C, D, R, Init_X, Init_Cov, Theta, Bias, X_update, X_smoothed
%

%% check some default values!
if ~isfield(handles,'switch_nondiagQ')
    handles.switch_nondiagQ = 0;
    display('switch nondiagQ is set as its default value to 0 --> in the inner function')
end
if ~isfield(handles,'switch_biasobs')
    handles.switch_biasobs = 0;
    display('switch biasobs is set as its default value to 0 --> in the inner function')
end

if ~isfield(handles,'init_type')
    handles.init_type = 'random';
    display('init_type is set as its default value to random --> in the inner function')
end
%% extract from handles
dim_hid = handles.dim_hid;
scale_dif_inp = handles.scale_dif_inp;
delta_inp = handles.delta_inp;
num_iter = handles.num_iter;
switch_nondiagQ = handles.switch_nondiagQ;
switch_nondiag = handles.switch_nondiag;
init_type = handles.init_type;
switch_biasobs = handles.switch_biasobs;%% build settings
if strcmp(handles.save_iter,'all')
    save_iter_number = 1:num_iter;
elseif strcmp(handles.save_iter,'startandend')
    save_iter_number = [1,num_iter];
end
spike_bs_init = handles.spike_bs_init;
%% set the settings, will also be used in decoder and smoother and maximizer
settings = struct;
%[settings.dim,~] = dim_Object;% dimension of states
settings.dim_input = 2; % the implementation of input is missing, arbitrarily we put input dim as 2
settings.dim_st = dim_hid; % dimension of latent state
settings.T = size(Y_Obs,2);%number of samples
settings.dim_Y = size(Y_Obs,1);% # of linear/ECoG Channels
settings.N = size(N_Obs,1);% nummber of neurons
settings.Scale_dif = scale_dif_inp; % scale difference in ourr multiscale measurement #spikes/linear obs
settings.delta = delta_inp; % time step
settings.it = num_iter; % number of iterations
settings.Input = zeros(settings.dim_input,settings.T);
settings.switch_nondiagQ = switch_nondiagQ;
settings.switch_biasobs = switch_biasobs;
%% initialization
%%%%% STEP2: INITIALIZATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%here we initialized the parameters to be fitted
if strcmp(init_type,'random')
    %Initialization.A = diag(random('Normal',0.6,0.1,settings.dim_st,1));
    % This is a good set of initialization, you could change depending on
    % your need
    Initialization.A = eye(settings.dim_st);
    Initialization.Q = diag(random('Uniform',0.025,0.07,1,settings.dim_st));
    Initialization.C = (1)*random('Normal',2,0.4,settings.dim_Y,settings.dim_st);
    Initialization.R = diag(random('Uniform',0.1,0.5,1,settings.dim_Y));
elseif strcmp(init_type,'subspace')
    [Initialization.A,~,Initialization.C,~,~,~,~,~,Initialization.Q,Initialization.R,Ss] = subid(Y_Obs,[],dim_hid-3,dim_hid); % third dimension is horizon
    Initialization.A = real(Initialization.A^(1/scale_dif_inp)); % to change the to the dynamics time scale
end

Initialization.B = zeros(settings.dim_st,settings.dim_input);
Initialization.D = zeros(settings.dim_Y,settings.dim_input);
Initialization.Init_X = zeros(settings.dim_st,1);
Initialization.Init_Cov = 0.001*eye(settings.dim_st,settings.dim_st);
Initialization.Theta = NaN(settings.dim_st+1,settings.N);

if strcmp(spike_bs_init,'meanFR')
    Initialization.Theta(1,:) = log(mean(N_Obs,2)/settings.delta);
elseif strcmp(spike_bs_init,'random')
    Initialization.Theta(1,:) = 1.5*ones(1,settings.N);
end
Initialization.Theta(2:settings.dim_st+1,:) = random('Normal',0.007,0.0015,settings.dim_st,settings.N);
Initialization.Bias=zeros(settings.dim_Y,1);
% set the 3-dimensional tensor of parameters in time...
A(:,:,1) = Initialization.A;
B(:,:,1) = Initialization.B;
Q(:,:,1) = Initialization.Q;
C(:,:,1) = Initialization.C;
D(:,:,1) = Initialization.D;
Init_X(:,1) = Initialization.Init_X;
Init_Cov(:,:,1) = Initialization.Init_Cov;
R(:,:,1) = Initialization.R;
Theta(:,:,1) = Initialization.Theta;
Bias(:,1) = Initialization.Bias;
%% starts EM
%%%%%% STEP4:EM ALGORITHM%%%%%%%
fprintf('EM algorithm starts \n');
% multiscale decoder
[Xupd_t_next,Xpred_t_next,Covupd_t_next,Covpred_t_next] = Decoder(A(:,:,1),B(:,:,1),Q(:,:,1),Init_X(:,1),Init_Cov(:,:,1),C(:,:,1),D(:,:,1),R(:,:,1),Theta(:,:,1),Y_Obs-repmat(Bias(:,1),1,settings.T),N_Obs,settings);
% fixed interval smoother
[ Xsmth_t_next,Covsmth_t_next,VarW_t_next,CorW_t_next ] = FIS_modified( Xupd_t_next,Xpred_t_next,Covupd_t_next,Covpred_t_next,A(:,:,1),settings );
%% build ITER struct
ITER = struct;
ITER.iter = struct;
ITER.iter(1).A = A(:,:,1);
ITER.iter(1).B = B(:,:,1);
ITER.iter(1).Q = Q(:,:,1);
ITER.iter(1).C = C(:,:,1);
ITER.iter(1).Init_X = Init_X(:,1);
ITER.iter(1).Init_Cov = Init_Cov(:,:,1);
ITER.iter(1).R = R(:,:,1);
ITER.iter(1).Bias = Bias(:,1);
ITER.iter(1).theta = Theta(:,:,1);
ITER.iter(1).X_smoothed = Xsmth_t_next;
ITER.iter(1).x_update = Xupd_t_next;

for i=2:settings.it
    fprintf('iteration %d starts \n',i);
    %% Maximization Step
    if switch_nondiag == 0
        [ A(:,:,i),B(:,:,i),Q(:,:,i),Init_X(:,i),Init_Cov(:,:,i),C(:,:,i),Bias(:,i),D(:,:,i),R(:,:,i),Theta(:,:,i) ] = Maximization_diag_bias( Xsmth_t_next,Covsmth_t_next,VarW_t_next,CorW_t_next,Y_Obs,N_Obs,Theta(:,:,i-1),settings);
    elseif switch_nondiag == 0
        [ A(:,:,i),B(:,:,i),Q(:,:,i),Init_X(:,i),Init_Cov(:,:,i),C(:,:,i),Bias(:,i),D(:,:,i),R(:,:,i),Theta(:,:,i) ] = Maximization_nondiag_bias( Xsmth_t_next,Covsmth_t_next,VarW_t_next,CorW_t_next,Y_Obs,N_Obs,Theta(:,:,i-1),settings);
    end
    %% Expectation Step
    % multiscale filter
    [Xupd_t_next,Xpred_t_next,Covupd_t_next,Covpred_t_next] = Decoder(A(:,:,i),B(:,:,i),Q(:,:,i),Init_X(:,i),Init_Cov(:,:,i),C(:,:,i),D(:,:,i),R(:,:,i),Theta(:,:,i),Y_Obs-repmat(Bias(:,i),1,settings.T),N_Obs,settings);
    % fixed interval smoother
    [ Xsmth_t_next,Covsmth_t_next,VarW_t_next,CorW_t_next ] = FIS_modified( Xupd_t_next,Xpred_t_next,Covupd_t_next,Covpred_t_next,A(:,:,i),settings );
    %% continue to build ITER struct
    ITER.iter(i).A = A(:,:,i);
    ITER.iter(i).B = B(:,:,i);
    ITER.iter(i).Q = Q(:,:,i);
    ITER.iter(i).C = C(:,:,i);
    ITER.iter(i).Init_X = Init_X(:,i);
    ITER.iter(i).Init_Cov = Init_Cov(:,:,i);
    ITER.iter(i).R = R(:,:,i);
    ITER.iter(i).Bias = Bias(:,i);
    ITER.iter(i).theta = Theta(:,:,i);
    if ismember(i,save_iter_number)
        ITER.iter(i).X_smoothed = Xsmth_t_next;
        ITER.iter(i).x_update = Xupd_t_next;
    end
end

%% saving results
results=struct;
% set the learned parameters
results.A = A(:,:,num_iter);
results.B = B(:,:,num_iter);
results.Q = Q(:,:,num_iter);
results.C = C(:,:,num_iter);
results.D = D(:,:,num_iter);
results.R = R(:,:,num_iter);
results.Init_X = Init_X(:,num_iter);
results.Init_Cov = Init_Cov(:,:,num_iter);
results.Theta = Theta(:,:,num_iter);
results.Bias=Bias(:,num_iter);
% get the inferred states
[Xupd_t_next,Xpred_t_next,Covupd_t_next,Covpred_t_next] = Decoder(A(:,:,num_iter),B(:,:,num_iter),Q(:,:,num_iter),Init_X(:,num_iter),Init_Cov(:,:,num_iter),C(:,:,num_iter),D(:,:,num_iter),R(:,:,num_iter),Theta(:,:,num_iter),Y_Obs-repmat(Bias(:,num_iter),1,settings.T),N_Obs,settings);
[ results.X_smoothed,~,~,~ ] = FIS_modified( Xupd_t_next,Xpred_t_next,Covupd_t_next,Covpred_t_next,A(:,:,num_iter),settings );
results.X_update = Xupd_t_next;

end

