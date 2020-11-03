function [ EIG,EIG_dis,modes ] = Generate_Poles_discreteplane( phandles )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function generate poles for the modes of the matrix A given some
% handles
%
% INPUTS: -phandles:
%            -phandles.freq : available frequencies
%            -phandles.drawing : 'random': gets from a uniform dist of
%            phandles.TC_range, 'fixed': from fixed values of phandles.decay
%            -phandles.TC_range : range of the time decay
%            -phandles.decayL available decays
% OUTPUTS: - EIG: vector of eigenvalues in real plane
%          - EIG: vecotr of eigenvalues in discrete plane
%          - modes: struct array contatining modes

%% get some values
freq = phandles.freq;
drawing = phandles.drawing;
delta = phandles.delta;

%% construct modes
modes = struct;
count_eig = 1;
for j = 1:length(freq)
    if freq(j) == 0
        modes(j).modetype = 'single';
        modes(j).eignum = count_eig;
        count_eig = count_eig+1;
    else
        modes(j).modetype = 'double';
        modes(j).eignum = [count_eig,count_eig+1];
        count_eig = count_eig+2;
    end
    if strcmp(drawing,'random')
        TC_range = phandles.TC_range;
        decay = random('Uniform',TC_range(1),TC_range(2));
    elseif strcmp(drawing,'fixed')
        decay = phandles.decay(j);
    end
    modes(j).decay = decay;
    modes(j).freq = freq(j);
    
    
end
%% discrete
EIG_dis = [];
for mode = 1:length(modes)
    if strcmp(modes(mode).modetype,'single')
        
        eig_val = 1/exp(delta/modes(mode).decay) ;
        
        EIG_dis = [EIG_dis,eig_val];
        
    elseif strcmp(modes(mode).modetype,'double')
        
        eig_val_norm = 1/exp(delta/modes(mode).decay) ;
        eig_val_angle = 2 * pi * delta * modes(mode).freq;
        
        eig_val_real = eig_val_norm * cos(eig_val_angle);
        eig_val_imag = eig_val_norm * sin(eig_val_angle);
        
        EIG_dis =[EIG_dis, ( eig_val_real + 1i* ( eig_val_imag ) )];
        EIG_dis =[EIG_dis,( eig_val_real - 1i* ( eig_val_imag ) )];
        
    end
    
end
%% construct continuous eigenvalues
EIG = ((EIG_dis)-1) / delta;
end

