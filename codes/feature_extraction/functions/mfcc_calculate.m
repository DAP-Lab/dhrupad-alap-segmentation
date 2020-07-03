function avg_MFCC = mfcc_calculate(x,fs,frm,hop,tex_win,tex_hop)

T_hop=tex_hop;  % Texture hop in s
T_frm=tex_win;  % Texture frame in s

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

mfcccoeff=zeros(noh-n_start,13);
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

avg_MFCC=zeros(Tnoh-Tn_start+1,13);
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

clear temp_MFCC mfcccoeff

end