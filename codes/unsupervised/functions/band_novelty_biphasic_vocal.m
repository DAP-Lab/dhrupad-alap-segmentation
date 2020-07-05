function  nov_band = band_novelty_biphasic_vocal2(sub_band_ener,mean_len)

%Log-Compression
compressionC=100;
comp_specData = log(1 + sub_band_ener.*compressionC)/(log(1+compressionC));
sub_band_ener=comp_specData;
%Smoothing filter
% biphasic function
n=-0.5:0.01:0.5;                                    %filter duration with samples at 0.01s
tu1=0.015; tu2=0.025; d1=0.02165; d2=-0.005;        % in sec


A=1/(tu1*sqrt(2*pi))*(exp(-((n-d1).^2)/(2*tu1^2)));
B=1/(tu2*sqrt(2*pi))*(exp(-((n-d2).^2)/(2*tu2^2)));
bi_ph=A-B;
filt_len=length(bi_ph);
filt_len = 2*round(filt_len./2)+1;

bandDiff = filter2(bi_ph, [repmat(sub_band_ener(:,1),1,floor(filt_len/2)),sub_band_ener,repmat(sub_band_ener(:,end),1,floor(filt_len/2))]);

bandDiff = bandDiff.*(bandDiff>0);
band_sm_fil = bandDiff(floor(filt_len/2):end-floor(filt_len/2)-1);

%Novelty of the band
novelty =  band_sm_fil;

% Normalize the band data
peak_nov=max(novelty);
novelty= novelty./peak_nov;
N_nov=length(novelty);
clear spec_data;
%Adaptive threshold
N_mn_frm=ceil(N_nov/mean_len);

nov_mn=zeros(1,N_mn_frm);k=0;
nov_z=[novelty zeros(1,mean_len)]; %clear novelty;
nov_adthr=zeros(mean_len,N_mn_frm);
for i_mn=1:N_mn_frm
  
   % Local mean
    nov_frm = nov_z(k+1:mean_len+k);
    loc_mn=mean(nov_frm);
    nov_mn(i_mn)=0.3*loc_mn;                        
    ad_thr=loc_mn;  
    
    nov_adthr_frm=(max(nov_frm,ad_thr));
    nov_adthr(:,i_mn)=transpose(nov_adthr_frm);
    k=k+mean_len;
end
clear nov_mn;
%}

nov_band=novelty(:)';