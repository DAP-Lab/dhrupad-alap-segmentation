function [acf_represent,fft_represent,lags]=acf_dft_represent2(onset_string,maxL,hop,tex_win,tex_hop)
    Ts_hop=tex_hop/hop;   % Number of frames in a Texture hop
    Ts_frm=tex_win/hop;   % Number of frames in a Texture frame
    
    % Nearest power of 2 to be taken as FFT size
    TFFT_size=2^(ceil(log2(Ts_frm))+1);
    
    % Calculate the number of hop according to the entered
    % hop and frame
    
    Tnoh=round((length(onset_string)-Ts_frm)/Ts_hop);
           
    % Calculation of hop from which the sample starts
        
    for n=1:Tnoh
        lwr_sample=n*Ts_hop-ceil(Ts_frm/2);
        if(lwr_sample>0)
         Tn_start=n;
         break;
        end
    end
       
     % Calculating ACF as well as DFT
    for nT=Tn_start:Tnoh
        
        lwr_sample=nT*Ts_hop-ceil(Ts_frm/2);
        uppr_sample=nT*Ts_hop+ceil(Ts_frm/2);
        
        clear temp_data;
        temp_data=onset_string(lwr_sample:uppr_sample);
        %%%%Energy normalization
        %temp_data=temp_data/sum(abs(temp_data).^2); 
        temp_data=temp_data/max(temp_data);     
        % Calculate the auto-correlation function based
        % representation
        
      
        % [tempacf, lags]=autocorr(temp_data,maxL);   % For Matlab14
        [tempacf, lags]=xcorr(temp_data,maxL,'coeff');
         tempacf=tempacf(1, 301:end);
         lags = lags(1, 301:end);
         
        for j=1:length(tempacf)-1
            acf_represent(j,nT)=tempacf(j);
        end
        Twin_fn=hann(length(temp_data));
         temp_data=Twin_fn.*temp_data';
        
        % Calculate the DFT based representation
        %Ttempfft=abs(fft(temp_data-mean(temp_data),TFFT_size));
        Ttempfft=abs(fft(temp_data-mean(temp_data),TFFT_size));
        Ttemp_fft_final=Ttempfft(1:TFFT_size/2);
       
         for j=1:length(Ttemp_fft_final)
            fft_represent(j,nT)=Ttemp_fft_final(j);
         end
      
    end

     