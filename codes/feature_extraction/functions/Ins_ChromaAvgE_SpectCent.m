function [avg_MFCC,avg_C,chroma_peakiness,avg_e,avg_speccentroid,Tnoh,Tnoh_chr] = Ins_ChromaAvgE_SpectCent(audio,fs,texture_hopsize,texture_winsize)
%%%%%%%%%%%%%%%  STEP 1 %%%%%%%%%%%%
%   Load the input file and get the acoustic features 
%    Chroma, Chroma variance, RMS energy and spectral centroid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 x1=audio;    
% Input the parameters

hop=0.01; % hop in s
frm=0.03; % frm in s
frm1=0.1; % frm in s for chroma computation


    T_hop=texture_hopsize;     % Texture hop in s
    T_frm=texture_winsize;      % Texture frame in s

    
%     pathinpop=strcat(filepath,fgetl(fidlist));
% wavpath=strcat(pathinpop,'.wav');
% %imagepath=strcat(pathinpop,'.jpeg');
% matpath=strcat(pathinpop,'_feature.mat');
    
  
siz=size(x1);
nos=siz(1);    % Number of samples in the file

%chksample=x1;
 %  hop and frm in samples
    s_hop=hop*fs;
    s_frm=frm*fs;
    
    s_frm1=frm1*fs;
    
% Nearest power of 2 to be taken as FFT size
    FFT_size=2^(ceil(log2(s_frm))+1);
    
    
% Calculate the number of hop according to the entered
    % hop and frame
        noh=round((nos-s_frm)/(s_hop));
        
%  Calculate the number of hop according to the entered hop and frame
     
        noh1=round((nos-s_frm1)/s_hop);
   
     
%clear n_start

% From which hop all the frames lie inside the file
    for n=1:noh
        lwr_sample=n*s_hop-ceil(s_frm/2);
        if(lwr_sample>0)
         n_start=n;
         break;
        end
    end
        
    
  %  Calculate from which noh1 all the frames will be lying inside the file
  
    for n=1:noh1
        lwr_sample=n*s_hop-ceil(s_frm1/2);
        if(lwr_sample>0)
         n_start1=n;
         break;
        end
    end
    
    % Define a windowing function
    sz_wind_fn=2*(ceil(s_frm/2))+1;
    win_fn=hann(sz_wind_fn);
    
    sz_wind_fn1=2*(ceil(s_frm1/2))+1;
    
    
    %   1601-> sz_winf_fn;
    
    win_fn1=hann(1601);
    
    
    
    %  Template for CHROMA function
       
    %wts=fft2chromamx(FFT_size1/2,12,fs,450);
    
    % Onsets
  
   
          
        %  FFT of the previous frame
        
    lwr_sample=(n_start1)*s_hop-ceil(s_frm/2);
    uppr_sample=(n_start1)*s_hop+ceil(s_frm/2);
    clear temp_data;
    temp_data=x1(lwr_sample:uppr_sample);

    % Do some operation eg. FFT in this case
    temp_data=temp_data.*win_fn;

    mag_fft=abs(fft(temp_data,FFT_size));
    mag_fft_prev=mag_fft(1:FFT_size/2);
       
    mfcccoeff=zeros(min(noh,noh1)-n_start1,13);
    spec_centroid=zeros(min(noh,noh1)-n_start1,1);
    e=zeros(min(noh,noh1)-n_start1,1);
    onset_string=zeros(min(noh,noh1)-n_start1,1);
    
    for n=n_start1+1:min(noh,noh1)
        
        %{
        % Read temp chroma from the file
        
        lwr_chroma=n*s_hop-ceil(s_frm1/2);
        uppr_chroma=n*s_hop+ceil(s_frm1/2);
        
        temp_chroma=x1(lwr_chroma:uppr_chroma);
        
        
         % FFT of the current frame in this case
        temp_chroma=temp_chroma.*win_fn1;
        
        mag_fft_chroma=abs(fft(temp_chroma,FFT_size1));
        mag_fft_currentchroma=mag_fft_chroma(1:FFT_size1/2);
        
        
        C(n,:)=mag_fft_currentchroma'*wts';
        
        % Normalization added
        C(n,:)=C(n,:)-mean(C(n,:));
       %}
        
        %{
        %  FFT of the previous frame
        
        lwr_sample=(n-1)*s_hop-ceil(s_frm/2);
        uppr_sample=(n-1)*s_hop+ceil(s_frm/2);
        clear temp_data;
        temp_data=x1(lwr_sample:uppr_sample);
        
        % Do some operation eg. FFT in this case
        temp_data=temp_data.*win_fn;
        
        mag_fft=abs(fft(temp_data,FFT_size));
        mag_fft_prev=mag_fft(1:FFT_size/2);
        %}
        if n>n_start1+1
            mag_fft_prev = mag_fft_current;
        end
        
        % Read the temp data from file
        lwr_sample=n*s_hop-ceil(s_frm/2);
        uppr_sample=n*s_hop+ceil(s_frm/2);
        clear temp_data;
        temp_data=x1(lwr_sample : uppr_sample);
        
        % Calculate the MFCC coefficient for the current data
        
        %mfcccoeff(n,:)=melfcc(temp_data,fs, 'maxfreq', 8000, 'numcep', 13, 'nbands', 40, 'fbtype', 'fcmel', 'dcttype', 1, 'usecmp', 1, 'wintime',frm, 'hoptime', hop, 'preemph', 0, 'dither', 1);
        mfcccoeff(n,:) = 0;
               
        % FFT of the current frame in this case
        temp_data=temp_data.*win_fn;
        
        mag_fft=abs(fft(temp_data,FFT_size));
        mag_fft_current=mag_fft(1:FFT_size/2);
        
        
        %%%% spectral centroid based distinction
        
         
         num=0;
     for j=1:length(mag_fft_current)
         num=num+j*mag_fft_current(j);
     end
     
     spec_centroid(n)=num/sum(mag_fft_current);
        
          
        
        %   calculate the flux function
         flux=mag_fft_current-mag_fft_prev;
         
        % Calculate short time energy 
        e(n)=sum(mag_fft_current);
         

        onset_string(n)=sum(flux);
         
        if(sum(flux)<0)
          onset_string(n)=0;
        end
         
         clear mag_fft; 
         
    end
        
   %    Texture Analysis of the features
       
   %%%%%%%%%%  TEXTURE ANALYIS 
   
   % Find the texture sampling frequency
    
    T_fs=(1/hop);     % in Hz
    
    
    Ts_hop=T_hop/hop;   % Number of samples in a Texture hop
    Ts_frm=T_frm/hop;   % Number of samples in a Texture frame
    
    
    % Nearest power of 2 to be taken as FFT size
    TFFT_size=2^(ceil(log2(Ts_frm))+1);
    
    
    % Calculate the number of hop according to the entered
    % hop and frame
    
        Tnoh=floor(round((length(onset_string)-Ts_frm)/Ts_hop));
        %Tnoh_chr=round((length( C)-Ts_frm)/Ts_hop);
        Tnoh_chr = 0;
        
        % Calculation of hop from which the sample starts
        
         for n=1:Tnoh
        lwr_sample=n*Ts_hop-ceil(Ts_frm/2);
        if(lwr_sample>0)
         Tn_start=n;
         break;
        end
         end
       
  % Calculating ACF vector as well as DFT representaton
      avg_e=zeros(Tnoh-Tn_start+1,1);
      var_e=zeros(Tnoh-Tn_start+1,1);
      avg_MFCC=zeros(Tnoh-Tn_start+1,13);
      avg_speccentroid=zeros(Tnoh-Tn_start+1,1);
      
      for nT=Tn_start:Tnoh
          lwr_sample=nT*Ts_hop-ceil(Ts_frm/2);
          uppr_sample=nT*Ts_hop+ceil(Ts_frm/2);
      clear temp_data;
        
        temp_data=onset_string(lwr_sample:uppr_sample);
        temp_e=e(lwr_sample:uppr_sample);
        temp_MFCC=mfcccoeff(lwr_sample:uppr_sample,:);
        temp_centroid=spec_centroid(lwr_sample:uppr_sample);
        %temp_C=C(lwr_sample:uppr_sample,:);
        
         avg_e(nT,:)=mean(temp_e);
         var_e(nT,:)=var(temp_e);
         avg_MFCC(nT,:)=mean(temp_MFCC);
         
         %avg_C(nT,:)=mean(temp_C);
         
         avg_speccentroid(nT,:)=mean(temp_centroid);
         
        %{  
        [tempacf lagop]=autocorr(temp_data,length(temp_data)-1);
        
        for j=1:length(temp_data)-1
            acf_represent(j,nT)=tempacf(j);
        end
    
        
        Twin_fn=hann(length(temp_data));
         temp_data=Twin_fn.*temp_data';
         temp_data=temp_data/sum(temp_data.^2);
       % Calculate the DFT based representation
        Ttempfft=abs(fft(temp_data,TFFT_size));
        Ttemp_fft_final=Ttempfft(1:TFFT_size/2);
        
        
         for j=1:length(Ttemp_fft_final)
            fft_represent(j,nT)=Ttemp_fft_final(j);
         end
        %}
         
         
         
        %temp_C=C(lwr_sample:uppr_sample,:);
        
        
        %avg_C(nT,:)=mean(temp_C);
        
    
      end
    

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   FEATURE POST PROCESSING
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %avg_C(1:10,:)=avg_C(11:20,:);
    avg_MFCC(1:10,:)=avg_MFCC(11:20,:);
    avg_e(1:10)=avg_e(11:20);
        
    avg_speccentroid(1:10)=avg_speccentroid(11:20);
    
  
   %chroma_peakiness=var(avg_C');
   %chroma_peakiness=chroma_peakiness';
   avg_C =0 ;
   chroma_peakiness = 0;
   
    
end 