% a light demonstration of how to use multiscale EM algorithm
% this script first generates an spontaneopus state-space model
% it then learns the parameters with multiscale EM algorithm
% and finally visualize the learned eigenvalues
%% OPTIONS for generating a spontaneous state-space model
OPTIONS = struct;
OPTIONS.dim_hid = 8;% dim of hidden state
OPTIONS.dim_hid_fit = 8;% dim of hidden state for fitting (this can be different from what we simulate data with)
OPTIONS.dim_inp = 2;% dim of input (arbitrary)
OPTIONS.dim_Y = 150;% number of LFP features
OPTIONS.dim_N = 30;% nummber of neurons
OPTIONS.scale_dif = 5;% scale difference between spike and lfp
OPTIONS.iteration_stop = 100 ;% number of iterations to run EM for
OPTIONS.delta = 0.002;% time step
OPTIONS.scale = 'MS';% generate multiscale observations
OPTIONS.Ttrain = 50000; % number of samples in training set
OPTIONS.spike_modulation = 'modesonly'; % do not change this
OPTIONS.normalization_c = 'peaktopeak'; %'std' or 'peaktopeak' or 'version1' 'predefined' %this is to determine how i normalize matrix C
OPTIONS.switch_nondiag = 0; % some settings for EM (explained in EM_multiscale_unsupervised_function.m)
OPTIONS.switch_nondiagQ = 0;% some settings for EM (explained in EM_multiscale_unsupervised_function.m)
OPTIONS.switch_biasobs = 0;% some settings for EM (explained in EM_multiscale_unsupervised_function.m)
OPTIONS.init_type = 'random'; %'subspace' or 'random', % some settings for EM (explained in EM_multiscale_unsupervised_function.m)
OPTIONS.save_iter = 'startandend';% some settings for EM (explained in EM_multiscale_unsupervised_function.m)
OPTIONS.spike_bs_init = 'random';% 'random' or 'meanFR', % some settings for EM (explained in EM_multiscale_unsupervised_function.m)
OPTIONS.number_systems = 1; % always set to 1
%% generate (or load) a sample state space model
[ REALIZ, TRUE ] = generate_spike_field_from_statespace( OPTIONS );
Y_train = REALIZ.Y_Obs;
N_train = REALIZ.N_Obs;
%% use EM algorithm which saves all parameters at iterations and also saves last step iteration parameters
handles = struct;
handles.dim_hid = OPTIONS.dim_hid_fit;
handles.scale_dif_inp = OPTIONS.scale_dif;
handles.delta_inp = OPTIONS.delta;
handles.num_iter = OPTIONS.iteration_stop;
handles.switch_nondiagQ = OPTIONS.switch_nondiagQ;
handles.switch_nondiag = OPTIONS.switch_nondiag;
handles.init_type = OPTIONS.init_type;
handles.switch_biasobs = OPTIONS.switch_biasobs;
handles.save_iter = OPTIONS.save_iter;
handles.spike_bs_init = OPTIONS.spike_bs_init;
%% fits the model with EM
[ resultsEM ,~,ITER ] = EM_multiscale_unsupervised_function( Y_train,N_train,handles);
%% plot learned eigs
fig = figure('Position',[50,248,783,752]);
S1 = scatter( real(eig( resultsEM.A )) , imag( eig(resultsEM.A)  ),100);hold on;
S1.MarkerEdgeColor = [76,0,153]/255;
S1.MarkerFaceColor = [76,0,153]/255;
S2 = scatter( real(eig( TRUE.A )) , imag(eig( TRUE.A )),100);
S2.MarkerEdgeColor = [0,255,255]/255;
S2.MarkerFaceColor = [0,255,255]/255;
legend([S1,S2],{'Learned modes','True modes'})
xlim([0.90,1]);
xticks((0.90:0.02:1));
ylim([0,0.08]);
yticks((0:0.02:0.08));
xlabel('Real')
ylabel('Imaginary')
title('learned modes vs. true modes')
set(gca,'view',[-55.9,90]);grid on
