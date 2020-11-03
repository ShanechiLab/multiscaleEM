function [x_esti,W_esti,ill_level] = KalmanAndPointProcessMultiscaleFilter(x_onestep,W_onestep,KF_sig,KF_mat,KF_noise,PPF_sig,PPF_para,PPF_delta,PPF_rate_fac)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb), Han Lin Hsieh and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a multiscale filter including continuous (linear model) and discrete (point process with log-linear firing rate) type of signals. The approximation 
% is based on the Lagrange approximation.
% 
%
% Inputs: 
% x_onestep:    The state vector (a column). Its formation is [num_state*1].
% W_onestep:    The corariance matrix of the state vector. Its form is [num_state*num_state].
% KF_sig:       The continuous signal from the linear model. It's a column vector [num_channel].
% KF_mat:       The observation matrix in the linear model the continuous signal. Its form is [num_channel*num_state].
% KF_noise:     The noise covariance at each channel. It can be a 2-dim matrix or a row vector. When it's a row vector, it's been considered as the diagonal of
%               the noise covariance matrix.
% PPF_sig:      A row vector [1*num_spike]. It records all spike values.
% PPF_para:     A 2-dim matrix [num_state+1*num_spike]. It's composed by all parameters in the point process with log-linear firing rate.  
% PPF_delta:    A scalar. It indicates the time step (real time) in the point process.
% PPF_rate_fac: A row vector [1*num_spike]. It contains the effect of the non-decoding values in the firing rate.
%
%
% Outputs:
% x_esti:       The updating state vector (a column). Its formation is [num_state*1].
% W_esti:       The updating corariance matrix of the state vector. Its form is [num_state*num_state].
% ill_level:    A scalar. The illness level of the W_esti.

% the illness level tolerance. "0" means no constraint.
ill_tole = 10^-12;

if(iscolumn(PPF_sig))
    PPF_sig = PPF_sig';
end

% the covariance matrix of KF
if(isrow(KF_noise))
    inv_KF_noise = diag(1./KF_noise);
else
    inv_KF_noise = inv(KF_noise);
end
KF_cov = KF_mat' * inv_KF_noise * KF_mat;

% the covariance matrix of PPF
% discard the constant part (first row) in the parameter matrix.
PPF_submatrix = PPF_para(2:end,:); 
num_state = size(PPF_submatrix,1);
weighted = exp( x_onestep' * PPF_submatrix + PPF_para(1,:) + PPF_rate_fac );
PPF_cov = ((ones(num_state,1) * weighted).*PPF_submatrix) * PPF_submatrix.'*PPF_delta;   % This is a summation process. Matrix can speed up the program.

% the mean vector of KF
KF_mean = KF_mat' * inv_KF_noise * (KF_sig - KF_mat * x_onestep);

% the mean vector of PPF
PPF_mean = PPF_submatrix * (PPF_sig - weighted * PPF_delta).';

% develop the x_esti and W_esti under the illness level criterion
W_esti = inv(W_onestep) + KF_cov + PPF_cov;
ill_level = rcond(W_esti);
if(ill_level>ill_tole)
    W_esti = inv(W_esti);
    x_esti = x_onestep + W_esti * (KF_mean + PPF_mean);
else
    W_esti = W_onestep;
    x_esti = x_onestep;
end

end