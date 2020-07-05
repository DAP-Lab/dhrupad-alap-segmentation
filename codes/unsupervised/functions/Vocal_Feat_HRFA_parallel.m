% Boundary detection and evaluation from extracted features

close all;


%% Arrays in which results will be stored for each version of each song separately
TP=[];
FA=[];
GT=[];

TP_post_overall=[];
FA_post_overall=[];
TP_avg_e_overall=[];
FA_avg_e_overall=[];
TP_spectCent_overall=[];
FA_spectCent_overall=[];

%% List of offset values used for data augmentation
audio_offset_list=[0 0.1 0.2 0.3 0.4];

%% Path to saved features
featpath = '../features/features_matlab';
song_data_path = '../../annotations/train_dataset.csv';

%% Loop over each song
for song_ind = 1:21

[titl, gt_orig] = read_song(song_data_path,song_ind);

%% Loop over each version of each song
for i=1:length(audio_offset_list)
%% Adjusting the ground truth
gt=gt_orig-audio_offset_list(i);
gt=(gt>0).*gt;    
    
%% Loading the features
allfeat=importdata(strcat(featpath,'/', titl,'_', num2str(audio_offset_list(i)))); % Tempo, Salience, Time, Energy, Spect Centroid
A = struct;
A.avg_MFCC = allfeat.avg_MFCC;
A.spectCent= allfeat.ste_sc_diff(:,2);
A.avg_e= allfeat.ste_sc_diff(:,1);
A.P = allfeat.rhythm_features;  %
clear allfeat

%% Parameters for SDM and peak-picking
A.kw = 50;
A.thresh = 0.3;
%for thresh = 0.1:0.05:0.8

%% Obtaining the novelty curves from SDM and then peak-picking
[A.post_sdm,A.nov_post]=sdm_nov(A.kw,'Euclidean',A.P);
A.nov_post=A.nov_post/max(A.nov_post);
A.peak_post=peak_pick_av(A.nov_post,40,40,30,A.thresh); %peak picking on novelty function corresponding to posterior features

[A.mfcc_sdm,A.nov_mfcc]=sdm_nov(50,'Euclidean',A.avg_MFCC);
A.nov_mfcc=A.nov_mfcc/max(A.nov_mfcc);
A.peak_mfcc=peak_pick_av(A.nov_mfcc,40,40,30,A.thresh); %peak picking on novelty function 
 
%% Peak picking for ST-energy and spectral centroid - using Vinutha's peak_pick_av
%A.peak_spectCent=peak_pick_av(A.spectCent,40,40,30,0.1); %peak picking on novelty function 
%[A.hrate_specCent, A.falarms_specCent] = hratefalarms(A.peak_spectCent, A.gt, 40);

%A.peak_avg_e=peak_pick_av(A.avg_e,40,40,30,0.1); %peak picking on novelty function 
%[A.hrate_specCent, A.falarms_specCent] = hratefalarms(A.peak_spectCent, A.gt, 40);

%% Peak picking for ST-energy and spectral centroid using Ualk's find_peak
[A.peak_avg_e, peak1_vala] = find_peak_2(-A.avg_e,6,40);
[A.peak_spectCent, peak2_vala] = find_peak_2(-A.spectCent,6,40); 

%% Calculating TP and FA for each method(Posterior, STE, SC) separately
[A.hrate_post, A.falarms_post] = hratefalarms(A.peak_post, gt, 40);  % True Hits & FAs
[A.hrate_avg_e, A.falarms_avg_e] = hratefalarms(A.peak_avg_e, gt, 40);  % True Hits & FAs
[A.hrate_spectCent, A.falarms_spectCent] = hratefalarms(A.peak_spectCent, gt, 40);  % True Hits & FAs

TP_post_overall=[TP_post_overall A.hrate_post];
FA_post_overall=[FA_post_overall A.falarms_post];

TP_avg_e_overall=[TP_avg_e_overall A.hrate_avg_e];
FA_avg_e_overall=[FA_avg_e_overall A.falarms_avg_e];

TP_spectCent_overall=[TP_spectCent_overall A.hrate_spectCent];
FA_spectCent_overall=[FA_spectCent_overall A.falarms_spectCent];

%% Combining all methods and then finding TP, FA
%peak_post2 = confidence_measure(A.peak_post,A.peak_mfcc,A.peak_spectCent,A.peak_avg_e,40);
peak_post2 = Combn_Pks_majority(A.peak_mfcc,A.peak_post,A.peak_spectCent,A.peak_avg_e,90);
[TP_comb, FA_comb] = hratefalarms(peak_post2, gt(gt~=0), 90);

%% Appending the result for each version of each song
TP=[TP TP_comb];
FA=[FA FA_comb];
gt_2 = gt(gt~=0);
GT=[GT length(gt_2)];

end



end
