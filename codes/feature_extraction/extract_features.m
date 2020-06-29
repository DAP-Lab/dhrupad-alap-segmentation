clear variables; clc
close all;

audio_dir='/media/Sharedata/rohit/DrupadAlapaudios/';

texture_hopsize=1; %seconds
texture_winsize=3; %seconds

pitch_shift_list = [0 1 2 3 4];
audio_offset_list = [0 0.1 0.2 0.3 0.4];

for song_ind = 1:20
parfor i_offset = 1:5
for pitch_shift = pitch_shift_list

audio_offset=audio_offset_list(i_offset);

A = struct;
[song_title, gt_boundaries] = read_song(song_ind);

fprintf('%s\t%d\t%f\t%f\t%f\n',song_title,song_ind,texture_winsize,audio_offset,pitch_shift);
fprintf('-----------------\n');

savedir=strcat('/media/Sharedata/rohit/drupad-alap-segmentation/features/features_from_matlab/', 'train_texwin', num2str(texture_winsize), '/');
if ~exist(savedir, 'dir')
       mkdir(savedir)
end

if audio_offset==0 && pitch_shift==0
    save_filename=strcat(savedir, song_title);
else
    save_filename=strcat(savedir, song_title, '_', num2str(audio_offset), '_', num2str(pitch_shift));
end

if pitch_shift==0
    audfile=strcat(audio_dir, strcat(song_title,'.wav'));
else
    audfile=strcat(audio_dir, 'pitch_shifted/', song_title, '_', num2str(pitch_shift), '.wav');
end
[x,fs]=audioread(audfile);

%% offset the audio for data augmentation
if audio_offset~=0
    x=x(audio_offset*fs:end);
end

%% parameters for spectral analysis
spect_win_sec=0.03;                             %window size in sec
spect_hop_sec=0.01;                             %hop size in sec
spect_win=spect_win_sec*fs;                     %in samples
spect_hop=spect_hop_sec*fs;                     %in samples

onsets  = syll_odf( x, spect_win, spect_hop, fs );
featureRate = ceil(fs/spect_hop);               %frame rate of Onsets

%% Rhythm features

% ACF computation
rhy_win = 20;                                   %texture window size in sec
rhy_stepsize = texture_hopsize;           %texture hop size in sec
rhy_win_len = round(rhy_win.*featureRate);      %in frames
rhy_stepsize=rhy_stepsize*featureRate;          %in frames

maxLag = 60/20;                                 %in sec, corresponding to 20 bpm
minLag = 60/600;                                %in sec, corresponding to 600 bpm
maxL=ceil(maxLag.*featureRate);                 %in frames
minL=ceil(minLag.*featureRate);                 %in frames

[acf_represent,fft_represent,lags] = acf_dft_represent2(onsets,maxL,texture_hopsize);

% Tempo, salience
[dft_acf_represent, tempo, sal, sal_acf] = tempo_sal_apr2015(acf_represent,fft_represent);

% Normalize
time_idx = 1:length(tempo);
A.scaled_features=[tempo',sal',time_idx'];
A.scaled_features=norm_feature(A.scaled_features,'c');

%% Posterior features by GMM modeling using BIC
options = statset('MaxIter',10000);
max_nmix = 6;
nlogl = cell(1,max_nmix);
P1 = cell(1,max_nmix);
likeli = Inf(1,max_nmix);
pen = Inf(1,max_nmix);
BIC = Inf(1,max_nmix);
BIC_comp = Inf(1,max_nmix);
obj = cell(1,max_nmix);

for k = 3:max_nmix
	rng('default');
    obj{k} = gmdistribution.fit(A.scaled_features,k,'Options',options,'Replicates',5,'Regularize',0.001,'CovType','diag');
    BIC(k)= obj{k}.BIC;
    [idx,nlogl{k},P1{k}] = cluster(obj{k},A.scaled_features);
    likeli(k)=nlogl{k};
    pen(k)=k*log(size(A.scaled_features,1));      %penalty
    BIC_comp(k)= nlogl{k}+50*pen(k);
end

[minBIC,ix] = min(BIC_comp);
numComponents = ix;
P1 = P1{numComponents};
A.P = P1;

%% Timbre features - STE and SC
[~,~,~,avg_e,spec_centroid] = Ins_ChromaAvgE_SpectCent(x,fs,texture_hopsize,texture_winsize);

avg_e_diff = biphasic_filt2(avg_e);
spec_centroid_diff = biphasic_filt2(spec_centroid);
avg_e_diff = -avg_e_diff;
spec_centroid_diff = -spec_centroid_diff;

% Normalize
avg_e_diff=norm_feature(avg_e_diff','c');
spec_centroid_diff=norm_feature(spec_centroid_diff','c');

%% Extract MFCCs
A.avg_MFCC=mfcc_calculate(x,fs,texture_winsize);

%% Save
len_features=min([size(A.scaled_features,1),size(avg_e_diff,1),size(A.avg_MFCC,1)]);

A.scaled_features=A.scaled_features(1:len_features,:);
avg_e_diff=avg_e_diff(1:len_features);
spec_centroid_diff=spec_centroid_diff(1:len_features);
A.avg_MFCC=A.avg_MFCC(1:len_features,:);
A.P=A.P(1:len_features,:);

A.scaled_features=[A.scaled_features, avg_e_diff, spec_centroid_diff];

parsave(save_filename, A, 'P','scaled_features','avg_MFCC','-mat')
end
end
end
