function [dft_acf_represent, tempo, sal, sal_acf,sl] = tempo_sal_2( acf_represent,fft_represent )

szf=size(fft_represent);

hop_acf=0.010;   %  Each lag in acf represent in s

bin_fft=50/szf(1); % bin in Hz


%Vectors to be multiplied must lie in the same range

%   Sensible pluck range is from 0.2 Hz to 13 Hz



sz=size(fft_represent);

mask_fft=zeros(1,sz(1));
multi_factor=zeros(1,sz(1));

new_represent=fft_represent;
lwr_fft=round(0.8*szf(1)/50);
uppr_fft=round(10*szf(1)/50);

for i=1:sz(1)
    if(i>lwr_fft&&i<uppr_fft)
        mask_fft(i)=1;
    else
        mask_fft(i)=0;
    end
    
end


for j=1:sz(2)  
tempfft=fft_represent(:,j);

tempfft=tempfft/max(tempfft);

% Use ACF to generate a matrix of freq and values that can be used
% for interpolation

%   Usage of v5cubic interpolation on the vector of interpolation strength

%  convert vectors to freq vs strength representation

    tempacf=acf_represent(:,j);
    
    xacf=1:length(tempacf);

    fxacf=1./(0.01*xacf);
    
    xm=1:szf(1);
    inp_multi=50/szf(1)*xm;
    
    multi_factor=interp1(fxacf,tempacf,inp_multi,'spline'); 
    
    
    for k=1:length(multi_factor)
        
        if(isnan(multi_factor(k))==1)
            multi_factor(k)=0;
        end
    end
    multi_factor_fft=multi_factor.*mask_fft;
    for ct=1:length(multi_factor_fft)
    multi_matrix(ct,j)=multi_factor_fft(ct);
    end
    
    
    new_fft=multi_factor_fft'.*tempfft;

          
     for k=1:length(new_fft)
         dft_acf_represent(k,j)=new_fft(k);
     end
end

 for i=1:sz(2)
temp1=dft_acf_represent(:,i);
temp2=multi_matrix(:,i);
[a, b]=max(temp1);
tempo(i)=b;
%sal(i)=a;
sal_acf(i)=temp2(b);
 end
sal=max(dft_acf_represent); 
sal_acf = sal_acf/max(sal_acf);
%sal = sal/max(sal);
%op=medfilt1(sal_acf,20);
for i=1:length(sal)
        if(isnan(sal(i))==1)
            sal(i)=0;
        end
end
op=medfilt1(sal,120);
for i=1:length(tempo)
if(op(i)<0.1)
%if(sal<0.1)
tempo(i)=0;
else
tempo(i)=tempo(i);
end
end

tempo=medfilt1(tempo,20);   

xf=1:length(tempo);


clear sl

Nf=20;
for i=Nf/2+1:length(tempo)-Nf/2
    clear tempo_data;
    clear x_data;
    tempo_data=tempo(i-Nf/2:i+Nf/2);
    x_data=xf(i-Nf/2:i+Nf/2);
    
    
    clear xnz
    clear ynz
     t=0;
     for k=1:length(tempo_data)
      if(tempo_data(k)~=0)    %%%%%%
         t=t+1;
         xnz(t)=x_data(k);
         ynz(t)=tempo_data(k);
      end                    %%%%%%%%
     end
     
     if(t>5)   
       p=polyfit(xnz,ynz,1);
       tempo_interp(i)=polyval(p,i);
       sl(i)=p(1);
     else
         sl(i)=0;
         tempo_interp(i)=0;
     end
       
end  
    
zm=zeros(1,10);
    zm=zm'; sl=sl';
    
    sl=[zm;sl];
    
    
    for i=1:length(sl)
        if(abs(sl(i))>5)
            sl(i)=0;
        end
    end
 
end
