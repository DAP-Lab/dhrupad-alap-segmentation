function case_study_analysis(song_ind, song_path)
close all;

addpath('./functions');
annotations_path='../../annotations/train_dataset.csv';

[song_name,gt]=read_song(annotations_path,song_ind);
[x,fs]=audioread(fullfile(song_path,strcat(song_name,'.wav')));

A=struct;

%% Averaging window size and hop
tex_win=3;
tex_hop=1;

%% %%%Parameters for spectral analysis
A.spect_win_sec=0.03;                   %window size in sec
A.hop_sec=0.01;                         %hop or stepsize in sec

A.spect_win=A.spect_win_sec*fs;           %win length in samples
A.hop=A.hop_sec*fs;                       %hop in samples   


A.onset_string  = syll_odf( x, A.spect_win,A.hop,fs );


A.featureRate = ceil(fs/A.hop);           % Feature rate of Onsets


%% %%%%%%%%%%%%%%%%%%%%%%Rhythmic Representation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Rhythmic feature- acf computed upto 3s lag
A.rhy_win = tex_win;                               %% rhythm window in sec
A.rhy_stepsize=tex_hop;                             %sec  

A.rhy_win_len = round(A.rhy_win.*A.featureRate);  %in samples
A.rhy_stepsize=A.rhy_stepsize*A.featureRate;      %in samples

A.maxLag = 60/20;                             % in sec, corresponding to 20 bpm
A.minLag = 60/600;                            % in sec, corresponding to 600 bpm

A.maxL=ceil(A.maxLag.*A.featureRate);             %corresponding to 20 bpm
A.minL=ceil(A.minLag.*A.featureRate);             %corresponding to 600 bpm

A.parameterRhym = [];
A.parameterRhym.win = A.rhy_win_len;
A.parameterRhym.stepsize= A.rhy_stepsize;
A.parameterRhym.maxL=A.maxL;
A.parameterRhym.minL=A.minL;


%% %%%%%%%%Display Rhythmogram
 
A.N_rhy_frm = fix((length(A.onset_string)-A.rhy_win_len+A.rhy_stepsize)/A.rhy_stepsize);

A.T_rhy= (1*A.rhy_stepsize:A.N_rhy_frm*A.rhy_stepsize)/A.featureRate;

 
%% %%%%%%%%%%%%%%%%%%%FFT Representation of rhythm 

[A.acf_represent,A.fft_represent,A.lags]=acf_dft_represent2(A.onset_string,A.maxL,tex_hop);  %%%%%%%%%%%%%%%PV's
 
A.lags_sec=A.lags/A.featureRate;

% Displaying Rhythmogram
figure()
imagesc(A.T_rhy,A.lags_sec,A.acf_represent);axis xy;colormap(flipud('gray'));title('Rhythmogram by ACF');
hold on;
for i=1:length(gt)
line([gt(i) gt(i)],[0 max(A.lags_sec)],'Color','w','LineStyle','--','LineWidth',2)
end
title('Rhythmogram by ACF'); xlabel('Time (s)'); ylabel('Lag (s)');
%}
%% %%%%%%%%%%%%%%%%%%%FFT Representation of rhythm 
A.NFFT=size(A.fft_represent,1);
A.F = (0:round(A.NFFT/2)-1)' / A.NFFT * A.featureRate;     %[0, fs/2) Hz 

%[A.dft_acf_represent, A.tempo, A.sal, A.sal_acf] = tempo_sal_jun2018( A.acf_represent,A.fft_represent);
[A.dft_acf_represent, A.tempo, A.sal, A.sal_acf] = tempo_sal_2( A.acf_represent,A.fft_represent);

%%  %%%%%%%%%%%% Display Tempo & Salience

%sal = medfilt1(A.sal,120);
sal_nor = A.sal./max(A.sal);
figure();
set(gcf,'Position',[100,100,700,405]);
plot(A.sal,'k'); %title('Salience');
%ylim([0 max(sal)]);
xlim([0 length(A.sal)]);
%set(gca,'FontSize',16,'FontWeight','bold');
hold on;
for i=1:length(gt)
line([gt(i) gt(i)],[0 max(A.sal)],'Color','black','LineStyle','--','LineWidth',2)
end
set(gca,'Fontsize',18);
xlabel('Time (s)','Fontsize',20); ylabel('Salience','Fontsize',20);


A.tempo_bpm=A.tempo*60*A.featureRate/(A.NFFT*2);

figure();
set(gcf,'Position',[100,100,700,405]);
plot(A.tempo_bpm,'.k'); %title('Tempo');
%ylim([0 1]);
xlim([0 length(A.tempo)]);
ylim ([0 round(max(A.tempo_bpm))]);
%set(gca,'FontSize',16,'FontWeight','bold');
hold on;
for i=1:length(gt)
line([gt(i) gt(i)],[0 round(max(A.tempo_bpm))],'Color','black','LineStyle','--','LineWidth',2)
end
set(gca,'Fontsize',18);
xlabel('Time (s)','Fontsize',20); ylabel('Tempo (BPM)','Fontsize',20);

%% %%%%%%%%%%%%%%%%%%%%%%%%Chroma Peakiness %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  

[A.mfcc,A.avg_C,A.chroma_peakiness,A.avg_e,A.spec_centroid] = Ins_ChromaAvgE_SpectCent(x,fs,tex_hop,3);


% save(matpath,'acf_represent','dft_acf_represent','ftempo','sal','sl','avg_e','chroma_peakiness','avg_C','avg_MFCC','avg_speccentroid','feat_process');

% figure(); plot(A.avg_e);
% axis([0 length(A.avg_e) 0 max(A.avg_e)])
% title('Short time Energy');
% hold on;
% for i=1:length(gt)
% line([gt(i) gt(i)],[0 max(A.avg_e)],'Color','g','LineStyle','--','LineWidth',2)
% end
% xlabel('Time (s)'); 


A.norm_spec_centroid = A.spec_centroid/max(A.spec_centroid);


A.spec_centroid_orig=A.spec_centroid;
A.avg_e_diff = biphasic_filt_timbre(A.avg_e,1,100);
A.spec_centroid=A.spec_centroid_orig;
A.spec_centroid(1:25)=0;
A.spec_centroid_diff = biphasic_filt_timbre(A.spec_centroid,1,100);
A.avg_e_diff = -A.avg_e_diff;
A.spec_centroid_diff = -A.spec_centroid_diff;

A.norm_avg_e_diff = A.avg_e_diff/max(A.avg_e_diff);
A.norm_spec_centroid_diff = A.spec_centroid_diff/max(A.spec_centroid_diff);
%A.avg_e_diff=norm_feature(A.avg_e_diff','c');
%A.spec_centroid_diff=norm_feature(A.spec_centroid_diff','c');

%% Plotting timbre features

figure(); plot(A.avg_e_diff, 'k');
set(gcf,'Position',[100,100,700,405]);
axis([0 length(A.avg_e_diff) min(A.avg_e_diff) max(A.avg_e_diff)])
hold on;
for i=1:length(gt)
line([gt(i) gt(i)],[min(A.avg_e_diff) max(A.avg_e_diff)],'Color','black','LineStyle','--','LineWidth',2)
end
set(gca,'Fontsize',18);
xlabel('Time (s)','Fontsize',20); ylabel('STE difference','Fontsize',20);

A.spec_centroid_diff_hz=A.spec_centroid_diff*(8000/513);

figure(); plot(A.spec_centroid_diff_hz, 'k');
set(gcf,'Position',[100,100,700,405]);
axis([0 length(A.spec_centroid_diff_hz) min(A.spec_centroid_diff_hz) 600])%max(A.spec_centroid_diff_hz)])
hold on;
for i=1:length(gt)
line([gt(i) gt(i)],[min(A.spec_centroid_diff_hz) 600],'Color','k','LineStyle','--','LineWidth',2)
end
set(gca,'Fontsize',18);
xlabel('Time (s)','Fontsize',20); ylabel('STC difference (Hz)','Fontsize',20);

% %feat_process=[tempo',sal',sl,avg_e,chroma_peakiness];
A.time = 1:length(A.tempo);
%A.feat_process=[A.tempo',A.sal'];
A.feat_process=[A.tempo',A.sal',A.time'];
A.feat_process1= [A.avg_e_diff', A.spec_centroid_diff'];

% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Do the normalization of the features
A.scaled_features=norm_feature(A.feat_process,'c');
A.scaled_features1=norm_feature(A.feat_process1,'a');


% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Posteriori feature computation on features without normalising
% %[P idx1]=posteriori_transformation(feat_process, 3);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
% %%%%%%%%%%%%%%%Posterior features by GMM modeling%%%%%%%%%%%%%%%%%%%%%%%%%%
% A.options = statset('MaxIter',10000);
% A.obj = gmdistribution.fit(A.feat_process,nMixture,'Options',A.options,'Replicates',5,'Regularize',0.001,'CovType','diag');
% % %obj=gmdistribution.fit(feature,3,'Replicates',5,'Options',options,'CovType','diag');
% % 
% [A.idx,A.nlogl,A.P] = cluster(A.obj,A.feat_process);

%% Posterior features by GMM modeling sing BIC
A.options = statset('MaxIter',10000);
max_nmix = 6;
nlogl = cell(1,max_nmix);
P1 = cell(1,max_nmix);
likeli = Inf(1,max_nmix);
pen = Inf(1,max_nmix);
A.BIC = Inf(1,max_nmix);
A.bic_comp = Inf(1,max_nmix);
A.obj = cell(1,max_nmix);

for k = 3:max_nmix
    rng('default');
    A.obj{k} = gmdistribution.fit(A.scaled_features,k,'Options',A.options,'Replicates',5,'Regularize',0.001,'CovType','diag');
    A.BIC(k)= A.obj{k}.BIC;
    [idx,nlogl{k},P1{k}] = cluster(A.obj{k},A.scaled_features);
    likeli(k)=nlogl{k};
    pen(k)=k*log(size(A.feat_process,1));      %%% Penalty
    A.bic_comp(k)= nlogl{k}+100*pen(k);
end


[minBIC,ix] = min(A.bic_comp);
A.numComponents = ix;
A.P1 = P1{A.numComponents};
A.P = A.P1;

fig=figure();
set(gca,'Position',[100,100,700,400]);
subplot(4,1,1)
plot(A.P(:,1),'.k'); xlim([1 length(A.P)]);
for i=1:length(gt)
 line([gt(i) gt(i)],[0 1],'Color','black','LineStyle','--','LineWidth',2)
end
set(gca,'XTick',[]);
yticks([0,0.5,1]);
yticklabels({'0.','','1.'});
set(gca,'FontSize',18);

subplot(4,1,2)
plot(A.P(:,3),'.k'); xlim([1 length(A.P)]);
for i=1:length(gt)
 line([gt(i) gt(i)],[0 1],'Color','k','LineStyle','--','LineWidth',2)
end
set(gca,'XTick',[]);
yticks([0,0.5,1]);
yticklabels({'0.','','1.'});
set(gca,'FontSize',18);

subplot(4,1,3)
plot(A.P(:,2), '.k'); xlim([1 length(A.P)]);
for i=1:length(gt)
 line([gt(i) gt(i)],[0 1],'Color','k','LineStyle','--','LineWidth',2)
end
set(gca,'XTick',[]);
yticks([0,0.5,1]);
yticklabels({'0.','','1.'});
set(gca,'FontSize',18);

subplot(4,1,4)
try
    plot(A.P(:,4), '.k'); xlim([1 length(A.P)]);
    for i=1:length(gt)
        line([gt(i) gt(i)],[0 1],'Color','k','LineStyle','--','LineWidth',2)
    end
catch
end

yticks([0,0.5,1]);
yticklabels({'0.','','1.'});
set(gca,'FontSize',18);
xlabel('Time (s)','Fontsize',20);

han=axes(fig,'Visible','off');
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel({'Posteriors',''},'Fontsize',20);

xl=xlabel('Time (s)','Fontsize',20);
xl.Position(2) = xl.Position(2) - 0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
 
% % SDM & novelty computation for posterior features

A.kw = 50;
A.thresh = 0.1;
n_peak =7;
thr_search=30;
%for thresh = 0.1:0.05:0.8
    [A.post_sdm,A.nov_post]=sdm_nov(A.kw,'Euclidean',A.P);
    A.nov_post=A.nov_post/max(A.nov_post);
    [A.peak_post, A.peak1_val_post] = find_peak_2(-A.nov_post,n_peak,thr_search);
    % 
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % % SDM & novelty computation for normalised rhythmic features
    [A.feat_sdm,A.nov_feat]=sdm_nov(A.kw,'Euclidean',A.scaled_features);
    A.nov_feat=A.nov_feat/max(A.nov_feat);
    [A.peak_feat, A.peak1_val_feat] = find_peak_2(-A.nov_feat,n_peak,thr_search); 
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 
    % % SDM & novelty computation for acf
    % 
    [A.acf_sdm,A.nov_acf]=sdm_nov(A.kw,'Euclidean',transpose(A.acf_represent));
    A.nov_acf=A.nov_acf/max(A.nov_acf);
    [A.peak_acf, A.peak1_val_acf] = find_peak_2(-A.nov_acf,n_peak,thr_search);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%% Extract MFCCs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A.avg_MFCC=mfcc_calculate(x,fs,A.spect_win_sec,A.hop_sec,tex_win,tex_hop);

len_features=min(size(A.scaled_features,1),size(A.scaled_features1,1));
A.scaled_features=[A.scaled_features(1:len_features,:) A.scaled_features1(1:len_features,:)];
if (length(A.avg_MFCC) < len_features)
    app_z= zeros(len_features-length(A.avg_MFCC), 13);
    A.avg_MFCC = [A.avg_MFCC; app_z];
else
A.avg_MFCC=A.avg_MFCC(1:len_features,:);
end
A.P=A.P(1:len_features,:);

%%%%%%%%%%%%%%%%%%%%%
thr_search=60;
n_peak =7;

[A.mfcc_sdm,A.nov_mfcc]=sdm_nov(50,'Euclidean',A.avg_MFCC);
A.nov_mfcc=A.nov_mfcc/max(A.nov_mfcc);

[A.peak_mfcc, A.peak1_vala] = find_peak_2(-A.nov_mfcc,n_peak,thr_search);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Displaying SDMs %%%%%%%%%%%%%%%%%

figure()
%subplot(1,4,1)
imagesc(A.acf_sdm);colormap('gray'); %title('Similarity matrix of ACF vectors');
axis([121 2810 121 2810]);
set(gca,'Fontsize',14);
set(gca,'TickLength',[.03 0]);
xticks([0 500 1500 2500])
xticklabels({'0','500','1500','2500'})
xlabel('Time (s)','fontsize',14);
%set(gca,'XTickLabel',[])
yticks([0 500 1500 2500])
yticklabels({'0','500','1500','2500'})
ylabel('Time (s)','fontsize',16);

figure()
%subplot(1,4,2)
imagesc(A.feat_sdm);colormap('gray'); %title('Similarity matrix of features');
axis([121 2810 121 2810]);
set(gca,'Fontsize',14);
set(gca,'TickLength',[.03 0]);
xticks([0 500 1500 2500])
xticklabels({'0','500','1500','2500'})
xlabel('Time (s)','fontsize',14);
%set(gca,'XTickLabel',[])
yticks([0 500 1500 2500])
yticklabels({'0','500','1500','2500'})
ylabel('Time (s)','fontsize',16);

figure()
%subplot(1,4,3)
imagesc(A.post_sdm);colormap('gray');  %title('Similarity matrix of posterior features');
axis([121 2810 121 2810]);
set(gca,'Fontsize',14);
set(gca,'TickLength',[.03 0]);
xticks([0 500 1500 2500])
xticklabels({'0','500','1500','2500'})
xlabel('Time (s)','fontsize',14);
%set(gca,'XTickLabel',[])
yticks([0 500 1500 2500])
yticklabels({'0','500','1500','2500'})
ylabel('Time (s)','fontsize',16);

figure()
%subplot(1,4,4)
imagesc(A.mfcc_sdm);colormap('gray'); %title('Similarity matrix of features');
axis([121 2810 121 2810]);
set(gca,'Fontsize',14);
set(gca,'TickLength',[.03 0]);
xticks([0 500 1500 2500])
xticklabels({'0','500','1500','2500'})
xlabel('Time (s)','fontsize',14);
%set(gca,'XTickLabel',[])
yticks([0 500 1500 2500])
yticklabels({'0','500','1500','2500'})
ylabel('Time (s)','fontsize',16);
%%%%%%%%%%%

%%%%%% Displaying Novs %%%%%%%%%%%

%tick_len=0.009;
figure()
subplot(4,1,1)
plot(A.nov_acf,'Color','black');
%hold on;
%plot(A.peak_acf(:,1),A.peak_acf(:,2),'Or');xlim([1 length(A.nov_acf)])
hold on;
for i=1:length(gt)
line([gt(i) gt(i)],[0 1],'Color','black','LineStyle','--','LineWidth',1.5)
end
set(gca,'Fontsize',10);
set(gca,'TickLength',[0.01 0]);
%xticks([0 500 1500 2500])
set(gca,'XTickLabel',[])
yticks([0 0.5 1])
yticklabels({'0','0.5','1'})
axis([121 2810 0 1]);
%xlabel('Time (s)','fontsize',10); 
%ylabel('(a)','fontsize',12);
title('(a)','fontsize',12);

%%%

subplot(4,1,2)
plot(A.nov_feat,'Color','black')
%hold on;
%plot(A.peak_feat(:,1),A.peak_feat(:,2),'Or');xlim([1 length(A.nov_feat)])
hold on;
for i=1:length(gt)
line([gt(i) gt(i)],[0 1],'Color','black','LineStyle','--','LineWidth',1.5)
end
set(gca,'Fontsize',10);
%set(gca,'YTickLabel',[])
set(gca,'TickLength',[.01 0]);
xticks([0 500 1500 2500])
%xticklabels({'0','500','1500','2500'})
set(gca,'XTickLabel',[])
yticklabels({'0','0.5','1'})
axis([121 2810 0 1]);
%xlabel('Time (s)','fontsize',10); %ylabel('Novelty Score');
axis([121 2810 0 1]);
%ylabel('(b)','fontsize',12);
title('(b)','fontsize',12);

%%%%%%

subplot(4,1,3)
plot(A.nov_post,'Color','black')
%hold on;
%plot(A.peak_post(:,1),A.peak_post(:,2),'Or');xlim([1 length(A.nov_feat)]);
hold on;
for i=1:length(gt)
line([gt(i) gt(i)],[0 1],'Color','black','LineStyle','--','LineWidth',1.5)
end
set(gca,'Fontsize',10);
set(gca,'TickLength',[.01 0]);
set(gca,'XTickLabel',[])
yticklabels({'0','0.5','1'})
axis([121 2810 0 1]);
%xlabel('Time (s)','fontsize',10); %ylabel('Novelty Score');
%set(gca,'YTickLabel',[])
axis([121 2810 0 1]);
%ylabel('(c)','fontsize',12);
title('(c)','fontsize',12);

%%%%%

subplot(4,1,4)
plot(A.nov_mfcc,'Color','black')
%hold on;
%plot(A.peak_post(:,1),A.peak_post(:,2),'Or');xlim([1 length(A.nov_feat)]);
hold on;
for i=1:length(gt)
line([gt(i) gt(i)],[0 1],'Color','black','LineStyle','--','LineWidth',1.5)
end
set(gca,'Fontsize',10);
set(gca,'TickLength',[.01 0]);
xticks([0 500 1500 2500])
xticklabels({'0','500','1500','2500'})
xlabel('Time (s)','fontsize',12); %ylabel('Novelty Score');
yticks([0 0.5 1])
yticklabels({'0','0.5','1'})
%set(gca,'YTickLabel',[])
axis([121 2810 0 1]);
%ylabel('(d)','fontsize',12);
title('(d)','fontsize',12);



end
