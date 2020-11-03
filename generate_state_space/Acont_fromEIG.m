function [ Acont ] = Acont_fromEIG( EIG )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this function computes Acont matrix from EIG (eigevnalues)
% INPUTS : (1) EIG : vector of all eigenvalues

% OUTPUTS : (1) Acont : A matrix in continuous time

dim = length(EIG);
Acont = zeros (dim,dim);
j = 1;
while j <=dim
    if imag(EIG(j)) == 0 % if eigenvalue is real
        Acont(j,j) = EIG(j);
        j = j+1;
    else
        Acont(j:j+1,j:j+1) = [real( EIG(j) ),imag( EIG(j) );-imag( EIG(j)),real( EIG(j) ) ];
        j = j+2;
    end
end

end

