function predict_boundaries(in_path)
% Boundary detection
close all;

%% List of offset values used for data augmentation
audio_offset_list=[0];

%% Path to saved features
featpath = '../../features/features_matlab';

%% Loop over each time-shifted version of test song
for i=1:length(audio_offset_list)

%% Loading the features
if isfile(in_path)
    filepath=in_path;
else
    filepath=strcat(featpath,'/', titl,'_', num2str(audio_offset_list(i)));
end

allfeat=importdata(filepath); % Tempo, Salience, Time, Energy, Spect Centroid
A = struct;
A.avg_MFCC = allfeat.avg_MFCC;
A.spectCent= allfeat.ste_sc_diff(:,2);
A.avg_e= allfeat.ste_sc_diff(:,1);
A.P = allfeat.rhythm_features;  %
clear allfeat

%% Parameters for SDM and peak-picking
A.kw = 50;
A.thresh = 0.3;

%% Obtaining the novelty curves from SDM and then peak-picking
[A.post_sdm,A.nov_post]=sdm_nov(A.kw,'Euclidean',A.P);
A.nov_post=A.nov_post/max(A.nov_post);
A.peak_post=find_peak_2(A.nov_post,6,40); %peak picking on novelty function corresponding to posterior features

[A.mfcc_sdm,A.nov_mfcc]=sdm_nov(50,'Euclidean',A.avg_MFCC);
A.nov_mfcc=A.nov_mfcc/max(A.nov_mfcc);
A.peak_mfcc=find_peak_2(A.nov_mfcc,6,40); %peak picking on novelty function 

%% Peak picking for ST-energy and spectral centroid
[A.peak_avg_e, ~] = find_peak_2(-A.avg_e,6,40);
[A.peak_spectCent, ~] = find_peak_2(-A.spectCent,6,40);

%% Combining all methods and then finding TP, FA
A.peak_post2 = Combn_Pks_majority(A.peak_mfcc,A.peak_post,A.peak_spectCent,A.peak_avg_e,90);

%% Output predicted boundaries
fprintf('Boundaries predicted by each method\n');
fprintf('\nRhythm:\n');
fprintf(' %d', A.peak_post);
fprintf('\nMFCC:\n');
fprintf(' %d', A.peak_mfcc);
fprintf('\nSpectral Centroid:\n');
fprintf(' %d', A.peak_spectCent);
fprintf('\nShort-time energy:\n');
fprintf(' %d', A.peak_avg_e);
fprintf('\nCombined:\n');
fprintf(' %d', A.peak_post2);
end
end
