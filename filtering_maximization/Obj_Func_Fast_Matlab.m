function [ value,varargout ] = Obj_Func_Fast_Matlab( a, Xsmth_t,Covsmth_t,N_Obs,delta,T,dim )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function obtains the objective function to estimate the best
% parameters of the spiking activity (spiking modulation matrix)
% for more details look at the paragraph right after equation (21) at 10.1109/TNSRE.2019.2913218
% the format in this function is compatible with fminunc
% INPUTS:
%         - a: represents theta
%         - Xsmth_t: smoothed values of states (non-causal inference) (dim * T)
%         - Covsmth_t: covariance of error of smoothed values of states (non-causal inference) (dim * dim * T)
%         - N_Obs: time series of spikes with size N*T, N is number of neurons
%         - delta: time step
%         - T: total number of observations: equal to size(N_Obs,2)
%         - dim: dimension of latent state, equals to size(Xsmth_t, 1)
% OUTPUTS:
%         outputs are compatible with the fminunc function (objective function and gradients)
%         - value: objective function value
%         - Genvector: derivative of objective fn with respect to spike params
%% calculate objective fn
theta = a;
aux_vec = reshape(Covsmth_t, dim, dim * T);
aux_vec2 = theta(2:dim + 1) * aux_vec;
aux_vec2prime = reshape(aux_vec2',dim,T)';
aux_vec3 = (aux_vec2prime * theta(2:dim + 1)')';
vec = (-N_Obs) .* (log(delta) + theta(1) + theta(2:dim + 1) * Xsmth_t) + delta * exp((theta(1) + theta(2:dim+1) * Xsmth_t + 0.5 * aux_vec3));
value=sum(vec); % value of objective function
%% calculates objective fn's deriative w.r.t spike params
if nargout>1
    % calculates gradients
    aux_vec1 = delta * exp(theta(1) + theta(2:dim + 1) * Xsmth_t + 0.5 * aux_vec3) * (Xsmth_t' + aux_vec2prime);
    aux_vec10 = zeros(1,dim);
    for i = 1:dim
        aux_vec10(i) = sum( N_Obs.*Xsmth_t(i,:));
    end
    vector = aux_vec1 - aux_vec10;
    Genvector = zeros(1,dim + 1);
    Genvector(2:dim + 1) = vector;
    
    aux_vec11 = delta * exp((theta(1) + theta(2:dim + 1) * Xsmth_t + 0.5 * aux_vec3)) - N_Obs;
    Genvector(1) = sum(aux_vec11);    
    varargout{1} = Genvector;
end

