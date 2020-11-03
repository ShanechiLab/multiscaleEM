function [ states ] = state_generator( A,B,Q,Input,init,T )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function generates a time series of states using 
% states(t+1) = A * states(t) + B * Input(t) + q(t)
% INPUTS:     (1)A: state transition matrix (dim, dim)
%             (2)B: the input-state matrix (dim, dim_input)
%             (3)Input: the input time series  (dim_input, T)
%             (2)Q: state noise covariance matrix (dim, dim)
%             (3)init: init has to be a cloumn vector, of initial state (dim, 1)
%             (4)T: number of time samples (1)
% 
% OUTPUTS:      (1)states: states time series (dim, T)

% if B is zero inoput is not important
%% 
if isequal(B,zeros(size(B)))   
   Input = zeros( size(B,2) , T );   
end

%% generate states
dim = size(A,1);
states(:,1) = init;
for i = 2:T
    states(:,i) = A * states(:,i-1) + + B * Input(:,i) + transpose(mvnrnd(zeros(dim,1),Q));
end

end

