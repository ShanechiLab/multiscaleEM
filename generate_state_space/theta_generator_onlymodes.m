function [ Theta_meanFR,Theta_modes ] = theta_generator_onlymodes( modes,N,type,handles )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2020 University of Southern California
% See full notice in LICENSE.md
% Hamidreza Abbaspourazad (@salarabb) and Maryam M. Shanechi
% Shanechi Lab, University of Southern California
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This function sets some random parameters for spike modulation
% INPUTS:     - modes: states time series
%             - N: number of neurons
%              - type: set to 'modesonly'
%              - handles:
%                 - modulation_depth_mean: mean of modulation vector of modes
%                 for different neurons
%                 - modulation_depth_std: std of modulation vector of modes
%                 for different neurons
%                 - modulation_mean_mean: mean of mean firing
%                 rate for different neurons
%                 - modulation_mean_std: std of mean firing
%                 rate for different neurons
%
% OUTPUT:     - Theta_meanFR: only first row will be used,i.e., mean FR, (dim + 1, N)
%             - Theta_modes: contains the
%                            modulation parameters, (dim, N)
% Theta can be derived as:
% Theta = zeros(dim,N);
% Theta(1,:) = Theta_meanFR(1,:);
% Theta(2:end,:) =  Theta_modes;
%% get handles
modulation_depth_mean = handles.modulation_depth_mean;
modulation_depth_std = handles.modulation_depth_std;

modulation_mean_mean =  handles.modulation_mean_mean;
modulation_mean_std = handles.modulation_mean_std;
%% get the number of modes
[dim_modes,T] = size(modes);
num_modes = dim_modes/2;
if dim_modes/2 - floor(dim_modes/2) ~=0
    error('modes need to come in double form!')
end
%% contribution
max_modes = zeros(num_modes,1);
for i=1:T
    for mode = 1:num_modes
        if norm(modes([(mode*2-1):mode*2],i))>max_modes(mode)
            max_modes(mode)=norm(modes([(mode*2-1):mode*2],i));
        end
    end
end

Theta_meanFR = NaN(1,N);
Theta_meanFR(1,:) = random('Normal',modulation_mean_mean,modulation_mean_std,1,N);% fo increasing centerout decoding performance
Theta_modes = zeros(dim_modes,N);

if strcmp(type,'modesonly')
    
    Nmodes = N;
    if Nmodes ~= 0
        %modes neurons
        mode_divider = (Nmodes-mod(Nmodes,num_modes))/num_modes;
        anglemode = linspace(0,2*pi,mode_divider);
        
        for i=1:Nmodes
            if dim_modes == 1
                v=random('Normal',0.9,0.1);
                Theta_modes(2,i) = v*nm;
                %Theta(2,i)=1.1559;
            else
                
                
                mode = ceil(i/mode_divider);
                if i > mode_divider * num_modes
                    mode =  num_modes;
                end
                whichangle = mod(i,mode_divider);
                if whichangle == 0
                    whichangle = mode_divider;
                end
                v =  [ cos(anglemode(whichangle)),sin(anglemode(whichangle)) ];
                
                contrib_mode = random('Normal',modulation_depth_mean,modulation_depth_std,1,1);
                mode_eff = v * modes([(mode*2-1):mode*2],:);
                
                max_mode_eff = 0;
                for t=1:T
                    if norm(mode_eff(:,t))>max_mode_eff
                        max_mode_eff=norm(mode_eff(:,t));
                    end
                    
                end
                nm = contrib_mode/max_mode_eff;
                
                Theta_modes([mode*2-1,mode*2],i) = (v)*nm;
                
                
            end
        end
        
    end
    
else
    error('no correct input type for the theta generator function')
end


end

