clear variables; clc;
close all;

% set(0, 'DefaultFigureVisible','off');
f_flags=fopen('flags_comb.txt', 'w');
fclose(f_flags);

KG_initPaths;
%HR = cell(20,4);
%FA = cell(20,4);
audio_offset_list=[0.1 0.2 0.3 0.4 0.5];

featpath1 = '../tta';                      % 3s for mfcc; 20s for all other feats
featpath3 = '../tta/tta_texwin3';          % for all other features
featpath20 = '../tta/tta_mfcc_texwin20';   % for all features

Recall_thr_post=[];
Recall_thr_avg_e=[];
Recall_thr_SpectCent=[];
Recall_thr_mfcc=[];
Recall_thr_comb=[];

Prec_thr_post=[];
Prec_thr_avg_e=[];
Prec_thr_SpectCent=[];
Prec_thr_mfcc=[];
Prec_thr_comb=[];



%for thresh = 0.1:0.05:0.8
for n_peak = 1:18

TP=[];   
FA=[];
GT=[];

TP_post_overall=[];
FA_post_overall=[];
TP_avg_e_overall=[];
FA_avg_e_overall=[];
TP_spectCent_overall=[];
FA_spectCent_overall=[];
TP_mfcc_overall=[];
FA_mfcc_overall=[];



for song_ind = 1:20
close all

[~, titl, gt_orig, ~ ] = tpv_rdsong(song_ind);

if strcmp(titl,'')
    continue
end
fprintf("%d, %d\n ", song_ind, n_peak );

% Loading the features
for i=1:5     %1:5
titl    

allfeat=importdata(strcat(featpath3,'/', titl,'_1_',num2str(audio_offset_list(i)))); % Tempo, Salience, Energy, Spect Centroid
feat= importdata(strcat(featpath1,'/', titl,'_1_',num2str(audio_offset_list(i)))); % mfcc

A = struct;
A.avg_MFCC = feat.avg_MFCC;  %3s tex win

A.spectCent= allfeat.scaled_features(:,5);

A.avg_e= allfeat.scaled_features(:,4);
A.P = feat.P;  
clear allfeat

gt=gt_orig-audio_offset_list(i);
gt=(gt>0).*gt;    

A.kw = 50;  % +/-50s
%A.kw = 25;  % +/-25s

%thr_search=60;
thr_search=30;

[A.post_sdm,A.nov_post]=sdm_nov(A.kw,'Euclidean',A.P);
%[A.post_sdm,A.nov_post]=sdm_nov(A.kw,'correlation',A.P);

A.nov_post=A.nov_post/max(A.nov_post);

[A.peak_post, peak1_vala] = find_peak_2(-A.nov_post,n_peak,thr_search);

[A.hrate_post, A.falarms_post] = hratefalarms(A.peak_post, gt, thr_search);  % True Hits & FAs

[A.mfcc_sdm,A.nov_mfcc]=sdm_nov(A.kw,'Euclidean',A.avg_MFCC);
%[A.mfcc_sdm,A.nov_mfcc]=sdm_nov(A.kw,'correlation',A.avg_MFCC);

 A.nov_mfcc=A.nov_mfcc/max(A.nov_mfcc);
 
 [A.peak_mfcc, peak1_vala] = find_peak_2(-A.nov_mfcc,n_peak,thr_search);
 
 [A.hrate_mfcc, A.falarms_mfcc] = hratefalarms(A.peak_mfcc, gt, thr_search);

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[A.peak_avg_e, peak1_vala] = find_peak_2(-A.avg_e,n_peak,thr_search);
[A.peak_spectCent, peak2_vala] = find_peak_2(-A.spectCent,n_peak,thr_search); 


[A.hrate_spectCent, A.falarms_spectCent] = hratefalarms(A.peak_spectCent, gt, thr_search);
[A.hrate_avg_e, A.falarms_avg_e] = hratefalarms(A.peak_avg_e, gt, thr_search);

TP_post_overall=[TP_post_overall A.hrate_post];
FA_post_overall=[FA_post_overall A.falarms_post];

TP_avg_e_overall=[TP_avg_e_overall A.hrate_avg_e];
FA_avg_e_overall=[FA_avg_e_overall A.falarms_avg_e];

TP_spectCent_overall=[TP_spectCent_overall A.hrate_spectCent];
FA_spectCent_overall=[FA_spectCent_overall A.falarms_spectCent];

TP_mfcc_overall=[TP_mfcc_overall A.hrate_mfcc];
FA_mfcc_overall=[FA_mfcc_overall A.falarms_mfcc];

% Combining the peaks noveltiesc & finding TP, FA 

peak_post2 = ombn_Pks_majority(A.peak_post,A.peak_mfcc,A.peak_spectCent,A.peak_avg_e,60);

[TP_comb, FA_comb] = hratefalarms(peak_post2, gt, thr_search);

TP=[TP TP_comb];
FA=[FA FA_comb];
gt_2 = gt(gt~=0);
GT=[GT length(gt_2)];

end

Recall_song_comb = TP./GT;
Detect_song_comb = TP +FA;
Prec_song_comb = TP./Detect_song_comb;
end

Recall_spectCent=sum(TP_spectCent_overall)/sum(GT);
Prec_spectCent= sum(TP_spectCent_overall)/sum([TP_spectCent_overall FA_spectCent_overall]);

Recall_avg_e=sum(TP_avg_e_overall)/sum(GT);
Prec_avg_e= sum(TP_avg_e_overall)/sum([TP_avg_e_overall FA_avg_e_overall]);

Recall_post=sum(TP_post_overall)/sum(GT);
Prec_post= sum(TP_post_overall)/sum([TP_post_overall FA_post_overall]);

Recall_mfcc=sum(TP_mfcc_overall)/sum(GT);
Prec_mfcc= sum(TP_mfcc_overall)/sum([TP_mfcc_overall FA_mfcc_overall]);

Recall_comb=sum(TP)/sum(GT);
Prec_comb= sum(TP)/sum([TP FA]);

Recall_thr_post=[Recall_thr_post Recall_post];
Recall_thr_avg_e=[Recall_thr_avg_e Recall_avg_e];
Recall_thr_SpectCent=[Recall_thr_SpectCent Recall_spectCent ];
Recall_thr_mfcc=[Recall_thr_mfcc Recall_mfcc];
Recall_thr_comb=[Recall_thr_comb Recall_comb ];
 
Prec_thr_post=[Prec_thr_post Prec_post];
Prec_thr_avg_e=[Prec_thr_avg_e Prec_avg_e];
Prec_thr_SpectCent=[Prec_thr_SpectCent Prec_spectCent];
Prec_thr_mfcc=[Prec_thr_mfcc Prec_mfcc];
Prec_thr_comb=[Prec_thr_comb Prec_comb];

end

FPR_thr_post=1-Prec_thr_post;
FPR_thr_avg_e= 1- Prec_thr_avg_e;
FPR_thr_SpectCent = 1-Prec_thr_SpectCent;
FPR_thr_mfcc = 1- Prec_thr_mfcc;
FPR_thr_comb =1-Prec_thr_comb;

% F-Score computation
F_scr_post= 2*Recall_thr_post.*Prec_thr_post./(Recall_thr_post+Prec_thr_post);
[max_f_scr_post, ind_p]=max(F_scr_post)
F_scr_SpectCent=2*Recall_thr_SpectCent.*Prec_thr_SpectCent./(Recall_thr_SpectCent+Prec_thr_SpectCent);
[max_f_scr_SC, ind_sc] = max(F_scr_SpectCent)
F_scr_STE=2*Recall_thr_avg_e.*Prec_thr_avg_e./(Recall_thr_avg_e+Prec_thr_avg_e);
[max_f_scr_STE, ind_ste]=max(F_scr_STE)
F_scr_mfcc=2*Recall_thr_mfcc.*Prec_thr_mfcc./(Recall_thr_mfcc+Prec_thr_mfcc);
[max_f_scr_mfcc, ind_mfcc]=max(F_scr_mfcc)
F_scr_comb=2*Recall_thr_comb.*Prec_thr_comb./(Recall_thr_comb+Prec_thr_comb);
[max_f_scr_comb, ind_comb]=max(F_scr_comb)
