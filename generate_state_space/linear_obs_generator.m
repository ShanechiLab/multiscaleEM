function [ Y_Obs ] = linear_obs_generator( states,C,D,R,input,bias )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generates linear observations from
% Y_Obs(t) = C * states(t) + D * input(t) + bias + r(t); cov(r(t)) = R
if isequal(D,zeros(size(D)))
    input = zeros( size(D,2) , size(states,2) );
end

Y_Obs = NaN(size(C,1),size(states,2));
dim_Y = size(C,1);
for i = 1: size(states,2)    
    Y_Obs(:,i) = C * states(:,i) + D*input(:,i) + transpose(mvnrnd(zeros(dim_Y,1),R)) + bias;    
end

end

