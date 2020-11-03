function [ Xupd_t,Xpred_t,Covupd_t,Covpred_t ] = Decoder(A,B,Q,Init_X,Init_Cov,C,D,R,Theta,Y_Obs,N_Obs,settings )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb), Han Lin Hsieh and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multiscale decoder for multiscale state space models as explained in this paper:
% 10.1088/1741-2552/aaeb1a
%
% In general, the states are infered with Bayesian inference and Laplace approximation
% the state space model is as follows:
%
% x_{t + 1} = A * x_{t} + q_t; COV(q_t) = Q
% y_{t} = C * x_{t} + r_t; COV(r_t) = R
% p(N_{t}|x_{t}) = (\lambda(x_{t})) ^ (N_{t}) * exp( -\lambda(x_{t}) * \Delta ); \lambda(x_{t}) = exp(-\alpha*x_{t} + \beta), \Delta: time-step
%
% INPUTS :
%         - A: state transition matrix (dim * dim)
%         - B: state-input matrix (default: zeros/not learned)  (dim * dim_inp)
%         - Q: state noise covariance matrix (dim * dim)
%         - Init_X: initial estimation of the latent state at t = 1  (dim * 1)
%         - Init_Cov: initial estimation of the latent state estimation error at t = 1  (dim * dim)
%         - C: observation emission matrix  (dim_Y * dim)
%         - D: observation-input matrix (default: zeros/not learned)  (dim_Y * dim_inp)
%         - R: observation noise covariance matrix (dim_Y * dim_Y)
%         - Theta: parameters of spike modulation -> [\beta_c;alpha_c] in each column for every neuron ( (dim + 1) * N)
%         - Y_Obs: the zero-meaned lfp observations (dim_Y * T)
%         - N_Obs: binary spiking observation (dim_Y * N)
%         - settings: struct with following fields:
%              - Scale_dif: difference in time-scale of spike and lfp:
%                           scale difference in spikes and LFPs, k, i.e., spikes are available at every time-step and lfp are available
%                           only at k, 2k, 3k, ...
%              - delta: timescale of dynamics, or sampling in seconds
%              - Input: input time-series
% OUTPUTS:
%         - X_upd_t: filtered values of states (causal inference) (dim * T)
%         - X_pred_t: filtered values of states (causal inference, prediction before update with observation at t) (dim * T)
%         - Covupd_t: covariance of error of filtered values of states (causal inference) (dim * dim * T)
%         - X_upd_t: covariance of error of filtered values of states (causal inference, prediction before update with observation at t) (dim *dim * T)

%% get some values
T = size(N_Obs,2);
[dim,~] = size(A);
Scale_dif = settings.Scale_dif;
delta = settings.delta;
%d_init=settings.d_init;
Input = settings.Input;

%% whitening lfp observations -> solve many numerical errors and is much faster
%this part is an optional for future i need to put a settings parameter
[eigvec_R,eigval_R,~] = svd(R);
eigval_R_vectorized = diag(eigval_R);

index_ill = find(eigval_R_vectorized<10^(-12),1);

eigval_R_new_vectorized = eigval_R_vectorized;
eigval_R_new_vectorized(index_ill:end) = [];
R_new = diag(eigval_R_new_vectorized);

transform_matrix = eigvec_R;
transform_matrix(:,index_ill:end) = [];

% whiten
R = R_new;
C = transform_matrix' * C;
Y_Obs = transform_matrix' * Y_Obs;
D = transform_matrix' * D;
%% start filtering
% set the place holders
Xupd_t = zeros(dim,T);
Xpred_t = zeros(dim,T);
Covupd_t = zeros(dim,dim,T);
Covpred_t = zeros(dim,dim,T);
% set the initial values
Xupd_t(:,1) = Init_X;
Covupd_t(:,:,1) = Init_Cov;

% run it iteratively
for i=2:T
    
    if (floor(i/Scale_dif) - i/Scale_dif) == 0
        % in this case multiscale filter will run because we have spike and lfp
        % predict
        [Xpred_t(:,i),Covpred_t(:,:,i)] = KalmanPrediction(A,B,Q,Xupd_t(:,i-1),squeeze(Covupd_t(:,:,i-1)),Input(:,i));
        % correct based on input
        Y_Obs_sub = Y_Obs(:,i) - D * Input(:,i);
        % update
        [Xupd_t(:,i),Covupd_t(:,:,i),ill_level] = KalmanAndPointProcessMultiscaleFilter(Xpred_t(:,i),squeeze(Covpred_t(:,:,i)),Y_Obs_sub,C,R,transpose(N_Obs(:,i)),Theta,delta,0);
    else
        % in this case only the point process filter will run
        % predict
        [Xpred_t(:,i),Covpred_t(:,:,i)] = KalmanPrediction(A,B,Q,Xupd_t(:,i-1),squeeze(Covupd_t(:,:,i-1)),Input(:,i));
        % update
        [Xupd_t(:,i),Covupd_t(:,:,i),ill_level] = PointProcessFilter(Xpred_t(:,i),squeeze(Covpred_t(:,:,i)),transpose(N_Obs(:,i)),Theta,delta);
    end
    
end

end

