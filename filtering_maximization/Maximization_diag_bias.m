function [ A,B,Q,Init_X,Init_Cov,C,bias,D,R,Theta ] = Maximization_diag_bias( Xsmth_t,Covsmth_t,VarW_t,CorW_t,Y_Obs,N_Obs,Previous_Theta,settings )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This code plays the role of maximization step in the multiscale EM
%algorithm, for the details read: 10.1109/TNSRE.2019.2913218
% INPUTS :
%         - Xsmth_t: smoothed values of states (non-causal inference) (dim * T)
%         - Covsmth_t: covariance of error of smoothed values of states (non-causal inference) (dim * dim * T)
%         - VarW_t: variance of latent states  (dim * dim * T)
%         - CorW_t: cross-covariance of latent states from adjacant steps (dim * dim * T-1)
%         - Y_Obs: time series of the LFPs, its dimension is (dim_Y*T), where dim_Y is the number of channles and T is
%         the total number of samples! if time-scale of LFP is k (available
%         at k,2k,3k,...), it is recommended to put the samples in between as NaN, however the model will never use them
%         - N_Obs: time series of spikes with size N*T, N is number of neurons
%         - Previous_Theta: spike modulation matrix from previous iteration, used as initial value of this iteration
%         - settings: consists of some settings, with following fields
%             - dim_input: arbitrary dimension of input
%             - scale_dif : scale difference in spikes and LFPs, k, i.e., spikes are available at every time-step and lfp are available
%                             only at k, 2k, 3k, ...
%             - delta :time step of the spike bins (or real time-scale of sampling/dynamics) (in seconds)
%             - Input: arbitrary input (usually zeros(dim_input, T))
%             -switch_nondiagQ: (Always put this as 1, for fast learning!) if it is 1 it means the state noise is non-diagonal, if it is 0 the state noise is diagonal
%             -switch_biasobs: whether to fit bias for linear observations, if 1, it learns a linear bias for Y_Obs
% OUTPUTS:
%         - A: state transition matrix
%         - B: state-input matrix (default: zeros/not learned)
%         - Q: state noise covariance matrix
%         - Init_X: initial estimation of the latent state at t = 1
%         - Init_Cov: initial estimation of the latent state estimation error at t = 1
%         - C: observation emission matrix
%         - bias: bias of lfp observation
%         - D: observation-input matrix (default: zeros/not learned)
%         - R: observation noise covariance matrix
%         - Theta: parameters of spike modulation -> [\beta_c;alpha_c] in each column for every neuron                    
%% get some values
[dim,T] = size(Xsmth_t);
dim_Y = size(Y_Obs, 1);
% optimization of B and D are not implemented
B = zeros(dim, settings.dim_input);
D = zeros(dim_Y, settings.dim_input);
% add some place holder
Xsmth_t_bias = [ones(1,settings.T);Xsmth_t];
% get dim_input
dim_input = settings.dim_input;
Scale_dif = settings.Scale_dif;
N = size(N_Obs, 1);
deltan = settings.delta;
Input = settings.Input;
Theta = zeros(1 + dim,N);
%% create some place holders (the main role of this place holders is to keep track of the sum of all inferrred values from expectation step in time)
Sum_VarW = zeros(dim,dim);
Sum_VarW_bias = zeros(dim + 1,dim + 1);
Sum_CorW = zeros(dim,dim);
Sum_CorYX = zeros(dim_Y,dim);
Sum_CorYX_bias = zeros(dim_Y,dim + 1);
Sum_VarY = zeros(dim_Y,dim_Y);
Sum_VarW_Scale = zeros(dim,dim);
Sum_VarW_Scale_bias = zeros(dim + 1,dim + 1);
Sum_Cor_inp2_Scale = zeros(dim_input,dim);
Sum_Cor_inpout = zeros(dim_Y,dim_input);
Sum_Cor_inp3_Scale = zeros(dim_input,dim_input);
Sum_Cor_inp=zeros(dim_input,dim);
Sum_Cor_inp2=zeros(dim_input,dim);
Sum_Cor_inp3=zeros(dim_input,dim_input);
VarW_t_bias=zeros(dim + 1,dim + 1,T);
%% start the sumations
for i=1:T
    
    VarW_t_bias(:,:,i)=[1,Xsmth_t(:,i)';Xsmth_t(:,i),VarW_t(:,:,i)];
    
    if floor(i/Scale_dif) - i/Scale_dif == 0
        Y_Obs_sub = Y_Obs(:,i) - D * Input(:,i);
        Sum_CorYX = Sum_CorYX + Y_Obs_sub * transpose(Xsmth_t(:,i));
        Sum_CorYX_bias = Sum_CorYX_bias + Y_Obs_sub * transpose(Xsmth_t_bias(:,i));
        Sum_VarW_Scale = Sum_VarW_Scale + VarW_t(:,:,i);
        Sum_VarW_Scale_bias = Sum_VarW_Scale_bias + VarW_t_bias(:,:,i);
        Sum_VarY = Sum_VarY + Y_Obs_sub * transpose(Y_Obs_sub);
        Sum_Cor_inp2_Scale = Sum_Cor_inp2_Scale + Input(:,i) * Xsmth_t(:,i)';
        Sum_Cor_inpout = Sum_Cor_inpout + Y_Obs(:,i) * Input(:,i)';
        Sum_Cor_inp3_Scale = Sum_Cor_inp3_Scale + Input(:,i) * (Input(:,i))';
    end
    if i==T
        Sum_VarW = Sum_VarW + VarW_t(:,:,i);
        Sum_VarW_bias = Sum_VarW_bias + VarW_t_bias(:,:,i);
    else
        Sum_VarW = Sum_VarW + VarW_t(:,:,i);
        Sum_VarW_bias = Sum_VarW_bias + VarW_t_bias(:,:,i);
        Sum_CorW = Sum_CorW + CorW_t(:,:,i); 
    end
    if i~=1
        Sum_Cor_inp = Sum_Cor_inp + Input(:,i) * Xsmth_t(:,i - 1)';
        Sum_Cor_inp2 = Sum_Cor_inp2 + Input(:,i) * Xsmth_t(:,i)';
        Sum_Cor_inp3 = Sum_Cor_inp3 + Input(:,i) * (Input(:,i))';
    end
end
%% optimize A
A = (Sum_CorW - B * Sum_Cor_inp) * inv(Sum_VarW - VarW_t(:,:,T));
%% optimize Q
Qaux = (Sum_VarW - VarW_t(:,:,1)) - A * transpose(Sum_CorW - B * Sum_Cor_inp) - (Sum_CorW - B * Sum_Cor_inp) * transpose(A) +...
       A * (Sum_VarW - VarW_t(:,:,T)) * transpose(A) - B * Sum_Cor_inp2 - (B * Sum_Cor_inp2)' + B * Sum_Cor_inp3 * B';

if settings.switch_nondiagQ == 0
    Inv_Q=zeros(dim,dim);
    for i=1:dim
        Inv_Q(i,i)=(T-1)/(Qaux(i,i));
    end
    Q=inv(Inv_Q);
elseif settings.switch_nondiagQ == 1
    Q = Qaux / (T-1);
end
%% optimize Init_X
Init_X=Xsmth_t(:,1);
%% optimize Init_Cov
Init_Cov=VarW_t(:,:,1)-Xsmth_t(:,1)*transpose(Xsmth_t(:,1));
%% optimize C and R
if settings.switch_biasobs == 1
    
    C_bias = Sum_CorYX_bias * inv(Sum_VarW_Scale_bias);
    
    Raux = Sum_VarY - C_bias * transpose(Sum_CorYX_bias) - Sum_CorYX_bias * transpose(C_bias)...
        + C_bias * Sum_VarW_Scale_bias * transpose(C_bias);
    
    Inv_R = zeros(dim_Y,dim_Y);
    for j = 1:dim_Y
        Inv_R(j,j)=(T/Scale_dif)/(Raux(j,j));
    end
    R=inv(Inv_R);
    
    C = C_bias(:,2:dim+1);
    bias = C_bias(:,1);
    
elseif settings.switch_biasobs == 0
    C = Sum_CorYX * inv(Sum_VarW_Scale);    
    Raux = Sum_VarY - C * transpose(Sum_CorYX) - Sum_CorYX * transpose(C) + C * Sum_VarW_Scale * transpose(C);
    Inv_R = zeros(dim_Y,dim_Y);
    for j = 1:dim_Y
        Inv_R(j,j) = (T/Scale_dif)/(Raux(j,j));
    end
    R = inv(Inv_R);
    
    C = C;
    bias = zeros(dim_Y,1);
end
%% optimize Theta
for i=1:N
    % numerical optimization given the cost fn derived at
    % Obj_Func_Fast_Matlab.m
    myf = @(a) Obj_Func_Fast_Matlab(a, Xsmth_t, Covsmth_t, N_Obs(i,:), deltan, T, dim);
    options = optimoptions(@fminunc,'Algorithm','trust-region','GradObj','on','Display','Iter');
    [temp,~,~,output_opt] = fminunc(myf, Previous_Theta(:,i)' , options);
    %fprintf('number of function counts : %d',output_opt.funcCount)
    Theta(:,i) = temp';

end


end
