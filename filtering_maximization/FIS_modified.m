function [ Xsmth_t,Covsmth_t,VarW_t,CorW_t ] = FIS_modified( Xupd_t,Xpred_t,Covupd_t,Covpred_t,A,settings )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function is the fast implementation of fixed interval smoother:
% for details read 10.1109/TNSRE.2019.2913218, equations (17)-(20)
% INPUTS :
%         inputs are outputs of Decoder.m, you can read there fore more info
%         - X_upd_t: filtered values of states (causal inference) (dim * T)
%         - X_pred_t: filtered values of states (causal inference, prediction before update with observation at t) (dim * T)
%         - Covupd_t: covariance of error of filtered values of states (causal inference) (dim * dim * T)
%         - X_upd_t: covariance of error of filtered values of states (causal inference, prediction before update with observation at t) (dim *dim * T)
%         - A: state transition matrix (dim * dim)
%         - settings: unused but for future use!
% OUTPUTS:
%         these
%         - Xsmth_t: smoothed values of states (non-causal inference) (dim * T)
%         - Covsmth_t: covariance of error of smoothed values of states (non-causal inference) (dim * dim * T)
%         - VarW_t: variance of latent states  (dim * dim * T)
%         - CorW_t: cross-covariance of latent states from adjacant steps (dim * dim * T-1)
%% get some values
[dim, T] = size(Xupd_t);
%% create place holders
Xsmth_t = zeros(dim, T);
Covsmth_t = zeros(dim, dim, T);
Varw_t = zeros(dim, dim, T);
CorW_t = zeros(dim, dim, T-1);

Xsmth_t(:, T) = Xupd_t(:, T);
Covsmth_t(:, :, T) = Covupd_t(:, :, T);
A_aux = zeros(dim,dim);

% start from t = T
VarW_t(:,:,T)=Covsmth_t(:,:,T)+Xsmth_t(:,T)*transpose(Xsmth_t(:,T));

for i=T-1:-1:1
    if rcond(Covpred_t(:,:,i+1))<10^(-12)
        A_aux=0;
    else
        % equation (12)
        A_aux=Covupd_t(:,:,i)*transpose(A)*inv((Covpred_t(:,:,i+1)));
    end
    % equation (13)
    Xsmth_t(:,i)=Xupd_t(:,i)+A_aux*(Xsmth_t(:,i+1)-Xpred_t(:,i+1));
    % equation (14)
    Covsmth_t(:,:,i)=Covupd_t(:,:,i)+A_aux*(Covsmth_t(:,:,i+1)-Covpred_t(:,:,i+1))*transpose(A_aux);
 
    auxCov=A_aux*Covsmth_t(:,:,i+1);
    % equation (19)
    VarW_t(:,:,i)=Covsmth_t(:,:,i) + Xsmth_t(:,i)*transpose(Xsmth_t(:,i));
    % equation (20)
    CorW_t(:,:,i)=auxCov' + Xsmth_t(:,i+1) * transpose(Xsmth_t(:,i));
end


end

