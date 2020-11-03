function [x_esti,W_esti,ill_level] = PointProcessFilter(x_onestep,W_onestep,N_t,parameter,delta)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb), Han Lin Hsieh and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function uses the discrete spike point process framework to update one-step predictions.
% The model of the discrete spike point process is a logarithm-linear model. ( log(firing rate)=parameter(1)+sum[x_onestep(i)*parameter(i+1)] ) 
%
% PLEASE NOTE THE FOLLOWING IMPORTANT THING: when the inverse condition is really too bad, the PPF won't update the x_onestep and W_onestep and will just keep
%   them as the outputs.
%
% about output: x_esti is the ML estimation of the state in the next step based on Gaussian approximation (it's a column vector)
%               W_esti is the ML estimation of the covariance matrix in the next step based on Gaussian approximation
%               ill_level is the condition number of "inv(W_onestep)+partial_sum", we want to check that whether it's good enough.
%
% about input : x_onestep, W_onestep are results of one-step estimation based on OFC model. (x_onestep is a column vector. Its length is n)
%               N_t is the spike signal. It's a vector with length "C". (it's a row vector)
%               parameter is a matrix recording all parameters for each neuron in the log linear model. It's dimension is (n+1)*C. Every column is a parameter
%                   set of one observe sequence. The first element is the basic firing rate and the rest are coefficients of the state variables.
%               delta is the mini time interval of each spike.

% Set up the ill condition tolerance.
ill_tole = 10^(-9);

% First, calculate the W_esti.
submatrix = parameter(2:end,:); % discard the constant part (first row) in the parameter matrix.
num_state = size(submatrix,1);

%inner_product_x = [1;x_onestep];   % for inner product sum in the exponential function (It's a col).
weighted = exp( x_onestep' * submatrix + parameter(1,:) );

partial_sum = ((ones(num_state,1) * weighted).*submatrix) * submatrix.'*delta;   % This is a summation process. Matrix can speed up the program.

if(rcond(W_onestep) > ill_tole)
    temp_W = inv(W_onestep) + partial_sum;
    ill_level = rcond(temp_W);
    % Now, check that whether the matrix's condition is good or not.
    if( ill_level > ill_tole )
        % now we solve the x_esti.
        W_esti = inv(temp_W);
        partial_sum = submatrix * (N_t - weighted * delta).';   % this one is the summation part of different neurons.
        x_esti = x_onestep + W_esti * partial_sum;
    else
        W_esti = W_onestep;
        x_esti = x_onestep;
        ill_level = rcond(W_esti);
    end
else
    W_esti = W_onestep;
    x_esti = x_onestep;
    ill_level = rcond(W_esti);
end
    
    
end
