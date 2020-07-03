function avg_MFCC = mfcc_calculate(x,fs,texture_win_size)

gpuDevice(2);

fprintf('Extracting MFCCs\n');
fprintf('-----------------\n');

hop=0.01; % hop in s
frm=0.03; % frm in s

T_hop=1;                % Texture hop in s
T_frm=texture_win_size; % Texture frame in s

siz=size(x);
nos=siz(1);    % Number of samples in the file
s_hop=hop*fs;
s_frm=frm*fs;
    
noh=round((nos-s_frm)/(s_hop));
        
% From which hop all the frames lie inside the file
for n=1:noh
    lwr_sample=n*s_hop-ceil(s_frm/2);
    if(lwr_sample>0)
        n_start=n;
    break;
    end
end

% Define a windowing function
% default - hamming

mfcccoeff=gpuArray(zeros(noh-n_start,13));
%mfcccoeff=zeros(noh-n_start,40);
for n=n_start+1:noh

    % Read the temp data from file
    lwr_sample=n*s_hop-ceil(s_frm/2);
    uppr_sample=n*s_hop+ceil(s_frm/2);
    temp_data=x(lwr_sample : uppr_sample);

    % Calculate the MFCC coefficient for the current data

    mfcccoeff(n,:)=melfcc(temp_data,fs, 'maxfreq', 8000, 'numcep', 13, 'nbands', 40, 'fbtype', 'fcmel', 'dcttype', 1, 'usecmp', 1, 'wintime',frm, 'hoptime', hop, 'preemph', 0, 'dither', 1);

end

Ts_hop=T_hop/hop;   % Number of samples in a Texture hop
Ts_frm=T_frm/hop;   % Number of samples in a Texture frame
sz = size(mfcccoeff,1);

Tnoh=floor(round(sz-Ts_frm)/Ts_hop);

% Calculation of hop from which the sample starts
 for n=1:Tnoh
    lwr_sample=n*Ts_hop-ceil(Ts_frm/2);
    if(lwr_sample>0)
         Tn_start=n;
         break;
    end
 end

avg_MFCC=gpuArray(zeros(Tnoh-Tn_start+1,13));
%avg_MFCC=zeros(Tnoh-Tn_start+1,40);
for nT=Tn_start:Tnoh
    lwr_sample=nT*Ts_hop-ceil(Ts_frm/2);
    uppr_sample=nT*Ts_hop+ceil(Ts_frm/2);

    temp_MFCC=mfcccoeff(lwr_sample:uppr_sample,:);
    avg_MFCC(nT,:)=mean(temp_MFCC);
end

avg_MFCC(1:10,:)=avg_MFCC(11:20,:);
avg_MFCC=gather(avg_MFCC);
temp_MFCC=gather(temp_MFCC);
mfcccoeff=gather(mfcccoeff);

%save(strcat('../saved_features/',titl,'_mfcc'), '-struct', 'A', 'avg_MFCC','-mat')

clear temp_MFCC mfcccoeff

end 

%%
% addpath('/home/asr-gpu/uddalok2/a/TPV_DrupadSegmentation/MatLab Routines/data6');
% for song_ind2=1:19
%     [~, titl, ~, ~] = tpv_rdsong(song_ind2);
%     load(strcat(titl,'mfcc.mat'));
%     clearvars title;
%     [mfcc_sdm,nov_mfcc]=sdm_nov(50,'Euclidean',avg_MFCC);
%     nov_mfcc=nov_mfcc/max(nov_mfcc);
%     peak_mfcc=peak_pick_av(nov_mfcc,40,40,30,0.1); %peak picking on novelty function corresponding to acf
%     
%     figure()
%     subplot(2,1,1)
%     imagesc(mfcc_sdm);colormap('gray');  title('Similarity matrix of posterior features');
%     %
%     subplot(2,1,2)
%     plot(nov_mfcc)
%     hold on;
%     plot(peak_mfcc(:,1),peak_mfcc(:,2),'Or');
%     hold on;
%     for i=1:length(gt)
%     line([gt(i) gt(i)],[0 1],'Color','g','LineStyle','--','LineWidth',2)
%     end
%     xlabel('Time (s)'); ylabel('Novelty Score');
%     %}
%     nov_acc{song_ind2} = nov_mfcc;
%     gt_acc{song_ind2} = gt;
%     
% end
% 
% [peak_mfcc,catgt] = catnovelty(nov_acc, gt_acc, 0, 0, 0.1, 90);
% [trueH_mfcc, falarms_mfcc] = roccal(peak_mfcc,catgt,90);
% load('data6/t.mat');
% t1 = max(t,length(peak_mfcc));
% 
% TPR_mfcc = trueH_mfcc/length(catgt);
% FPR_mfcc = falarms_mfcc/t1;
% [TPR_mfcc, FPR_mfcc] = conhull([TPR_mfcc 1], [FPR_mfcc 1]);
