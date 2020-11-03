function [x_onestep, W_onestep] = KalmanPrediction(A,B,W,x_pre,W_pre,Input)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb), Han Lin Hsieh and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% About output:
% x_onestep:    the one step prediction of the goal at the next step
% W_onestep:    the one step prediction about the covariance matrix at the next step
%
% About inputs :
% A,B,W:    they are parameters of the linear dynamical system ( x.t+1 = A*x.t+B*u.t+w.t )
% Q,R:      they are symetric matrix coefficients of the cost function ( J = Sigma_all_t_from_1_to_infinum( (<x.t,x.t>|Q)+(<u.t,u.t>|R) )
% L:        the optimal feedback control multiplier.
% x_pre:    the last time dynamic state
% W_pre:    the last time covariance matrix
% x_goal:   the final target of the whole process. Its dimension is the same as the x_pre.
%
% NOTE: In this program, we don't set the stopping time but just only give a one step ahead prediction. The whole process is finite or infinite horizon is depending
%       on the calling function, which inputs the corresponding L each time. The reason is that in finite horizon testing, the recursive formula is calculated
%       backward, which means that it must be calculated ahead, not on the real time.
%
% NOTE: Split the target terms from the state space model based on two reasons. First, this is a more natural way do dispaly the model. Everything has its own
%       state space, but not everyone know where to go; second, split target terms can make the whole simulation more flexible. It can change goal at any time
%       in the simulation process after start.

if(any(any(B)) == 0) % which means that there is no control term in this model and it's a random walk.
    
    x_onestep = A * x_pre;
    W_onestep = A * W_pre * A' + W;
    
else
    
    % Otherwise, calculate the one step prediction of the next time with the gain matrix.
    x_onestep = (A) * x_pre + B * Input;
    W_onestep = (A) * W_pre *(A)' + W;
    
end

end



