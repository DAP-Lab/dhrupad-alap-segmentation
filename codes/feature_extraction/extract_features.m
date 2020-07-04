function extract_features(in_path, varargin)

addpath('./functions/');

%% default parameters
tex_hop=1; %seconds
tex_win=3; %seconds
audio_offset_list = [0];
out_path=strcat('../../features/features_from_matlab/', 'texwin_', num2str(tex_win), '/');
in_dir=[];

%% parse input arguments
p = inputParser;
addRequired(p,'in_path');
addParameter(p,'in_dir',in_dir);
addParameter(p,'tex_win',tex_win);
addParameter(p,'tex_hop',tex_hop);
addParameter(p,'offsets',audio_offset_list);
addParameter(p,'out_path',out_path);
parse(p,in_path,varargin{:});

tex_hop=p.Results.tex_hop;
tex_win=p.Results.tex_win;
audio_offset_list = p.Results.offsets;
out_path=p.Results.out_path;
in_dir=p.Results.in_dir;

%% get list of songs to extract features for
if isfile(in_path) || isfile(fullfile(in_dir,in_path))
    if strcmp(in_path(end-2:end),'wav')
        songs_list=[];
        n_songs=1;
    elseif strcmp(in_path(end-2:end),'txt') || strcmp(in_path(end-2:end),'csv')
        try
            songs_list = readtable(in_path);
            songs_list = string(songs_list.ConcertName(:));
        catch
            fid=fopen(in_path,'r');
            songs_list=textscan(fid,'%s');
            songs_list=songs_list{1,1};
        end
        n_songs=length(songs_list);
    end
else
    errmsg=['Invalid input. Allowed formats - a single ".wav" file, or a list of audio files in ' ...
        '".csv", ".txt" format'];
    error(errmsg);
end

%% feature extraction part (loop over all songs)
for song_ind = 1:n_songs
    
    %% get song name and directory
    if isempty(songs_list)
        if isempty(in_dir)
            in_dir=strsplit(in_path,'/');
            song_title=in_dir{end};
            in_dir{end}='';
            in_dir=strjoin(in_dir,'/');
        else
            song_title=in_path;
        end
    else
        in_path = char(songs_list(song_ind));
        if isempty(in_dir)
            in_dir=strsplit(in_path);
            song_title=in_dir{end};
            in_dir{end}='';
            in_dir=strjoin(in_dir,'/');
        else
            song_title=in_path;
        end
    end

    fprintf('Extracting features for song no: %d\t%s\n',song_ind,song_title);

    for audio_offset=audio_offset_list
        fprintf('audio offset %1.3f\n',audio_offset);

        % structure to save feature values
        A = struct;
        if ~exist(out_path, 'dir')
               mkdir(out_path)
        end

        if audio_offset_list==[0]
            save_filename=fullfile(out_path, strrep(song_title,'.wav','.mat'));
        else
            save_filename=fullfile(out_path, strcat(strrep(song_title,'.wav',''), '_', num2str(audio_offset),'.mat'));
        end

        % load audio
        if ~strcmp(song_title(end-3:end),'.wav')
            song_title=strcat(song_title,'.wav');
        end
        [x,fs]=audioread(fullfile(in_dir,song_title));

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

        %% Rhythm features - Tempo, salience

        % ACF computation
        rhy_winsize = 20;                                %texture window size in sec
        rhy_stepsize = tex_hop;                          %texture hop size in sec
        %rhy_win_len = round(rhy_win.*featureRate);      %in frames
        %rhy_stepsize=rhy_stepsize*featureRate;          %in frames

        maxLag = 60/20;                                 %in sec, corresponding to 20 bpm
        minLag = 60/600;                                %in sec, corresponding to 600 bpm
        maxL=ceil(maxLag.*featureRate);                 %in frames
        %minL=ceil(minLag.*featureRate);                %in frames

        [acf_represent,fft_represent,~] = acf_dft_represent2(onsets,maxL,spect_hop_sec,rhy_winsize,rhy_stepsize);

        % Tempo, salience
        [~, tempo, sal, ~] = tempo_sal_apr2015(acf_represent,fft_represent);

        % Normalize
        time_idx = 1:length(tempo);
        A.tempo_sal=[tempo',sal',time_idx'];
        A.tempo_sal=norm_feature(A.tempo_sal,'c');

        %% Rhythm features - Posterior features by GMM modeling using BIC
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
            obj{k} = gmdistribution.fit(A.tempo_sal,k,'Options',options,'Replicates',5,'Regularize',0.001,'CovType','diag');
            BIC(k)= obj{k}.BIC;
            [idx,nlogl{k},P1{k}] = cluster(obj{k},A.tempo_sal);
            likeli(k)=nlogl{k};
            pen(k)=k*log(size(A.tempo_sal,1));      %penalty
            BIC_comp(k)= nlogl{k}+50*pen(k);
        end

        [minBIC,ix] = min(BIC_comp);
        numComponents = ix;
        P1 = P1{numComponents};
        A.rhythm_features = P1;

        %% Timbre features - STE and SC
        [~,~,~,avg_e,spec_centroid] = Ins_ChromaAvgE_SpectCent(x,fs,tex_hop,tex_win);

        avg_e_diff = biphasic_filt2(avg_e);
        spec_centroid_diff = biphasic_filt2(spec_centroid);
        avg_e_diff = -avg_e_diff;
        spec_centroid_diff = -spec_centroid_diff;

        % Normalize
        A.ste_sc_diff=[avg_e_diff',spec_centroid_diff'];
        A.ste_sc_diff=norm_feature(A.ste_sc_diff,'c');

        %% Timbre features - MFCCs
        A.avg_MFCC=mfcc_calculate(x,fs,spect_win_sec,spect_hop_sec,tex_win,tex_hop);

        %% Save
        len_features=min([size(A.tempo_sal,1),size(A.ste_sc_diff,1),size(A.avg_MFCC,1)]);
        A.tempo_sal = A.tempo_sal(1:len_features,:);
        A.ste_sc_diff=A.ste_sc_diff(1:len_features,:);
        A.avg_MFCC=A.avg_MFCC(1:len_features,:);
        A.rhythm_features=A.rhythm_features(1:len_features,:);

        parsave(save_filename, A, 'tempo_sal', 'rhythm_features', 'ste_sc_diff', 'avg_MFCC','-mat')
    end
end
end