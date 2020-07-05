function  nov_band = biphasic_filt_timbre(signal,~,winlen)


% biphasic function
n=-30:100;                                    %filter duration with samples 1s

if winlen==25
    tu1=17.5; tu2=7.5; d1=25; d2=-2.5;        % in sec
elseif winlen==50
    tu1=35; tu2=15; d1=50; d2=-5;        % in sec
elseif winlen==100
    tu1=70; tu2=30; d1=100; d2=-10;        % in sec
end



A=1/(tu1*sqrt(2*pi))*(exp(-((n-d1).^2)/(2*tu1^2)));
B=1/(tu2*sqrt(2*pi))*(exp(-((n-d2).^2)/(2*tu2^2)));
bi_ph=A-B;
filt_len=length(bi_ph);
filt_len = 2*round(filt_len./2)+1;

signaldiff = filter2(bi_ph, [repmat(signal(1),1,floor(filt_len/2)),signal',repmat(signal(end),1,floor(filt_len/2))]);

band_sm_fil = signaldiff(floor(filt_len/2):end-floor(filt_len/2)-1);
novelty =  band_sm_fil;
nov_band=novelty(:)';