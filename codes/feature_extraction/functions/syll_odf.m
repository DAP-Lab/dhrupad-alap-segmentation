function  [onsets, freq_spec] = syll_odf( audfile, ana_win,ana_hop,fs )
%Returns the syllabic-onsets of the audio 
x=audfile;
N_x=length(x); 

spect_win=ana_win;           %win length in samples
hop=ana_hop;                 %hop in samples   

 
win=hamming(spect_win);

% noh=round((nos-s_frm)/(s_hop));
%N_sf_frame=ceil(N_x/hop);
N_sf_frame=ceil((N_x-spect_win)/hop);
%%% Sub band energy

%%%%% Spectral content
x_z=([x' zeros(1,spect_win)]);
k = 0;
%clear x;
NFFT = 2.^nextpow2(spect_win);            % FFT length, next higher power of 2 
S_mat=zeros(NFFT/2,N_sf_frame);  

filt_pb1=floor((1*NFFT)/fs);filt_pb2=floor((8000*NFFT)/fs);   %Instrumental
%filt_pb1=floor((600*NFFT)/fs);filt_pb2=floor((2800*NFFT)/fs);   %Vocal band
sub_band_ener=zeros(1,N_sf_frame);

for i_sf=1:N_sf_frame
    frame = x_z(k+1:spect_win+k);
    win_frame=frame .* win';
    ener_bin_arr=zeros(1,NFFT/2);
    Y = fft(win_frame, NFFT);  
    for m=1:NFFT/2
        if m >= filt_pb1 && m <= filt_pb2
            ener_bin_arr(m)= abs(Y(m)).^2;
            freq_spec(m,i_sf) = 2*abs(Y(m));
        end
    end
    S_mat(:,i_sf)= transpose(Y(1:round(NFFT/2)));   %another half is just mirror
%         if(mod(i_sf,100))==0
%             i_sf
%         end
    k = k+hop;
    sub_band_ener(i_sf)=(sum(ener_bin_arr)); 
end

mean_len = 50;                                 %in sec
mean_len = max(ceil(mean_len*fs/hop),3);

onsets = band_novelty_biphasic_vocal2(sub_band_ener,mean_len);

end

