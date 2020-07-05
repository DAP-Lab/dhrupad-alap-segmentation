function [peak_out] = conf_measr_4mthds_var_refmthd_majortyPks(conf_peak1, conf_peak2, conf_peak3, conf_peak4, threshold)

%peak_in = conf_peak3; % SC as ref
%peaks1=conf_peak1; peaks2=conf_peak2; peaks3=conf_peak4;

peak_in = conf_peak2; % mfcc as ref 
peaks1=conf_peak1; peaks2=conf_peak3; peaks3=conf_peak4;

%peak_in = conf_peak1; % post as ref 
%peaks1=conf_peak3; peaks2=conf_peak2; peaks3=conf_peak4;

peak_out=[];
for i=1:length(peak_in)
    if (isempty(peaks1) || isempty(peaks2) || isempty(peaks3))
        break
    end
    
    peak_range=peak_in(i)-threshold/2:0.1:peak_in(i)+threshold/2;

    flag1='False';
    for j1=1:length(peaks1)
        if ismember(peaks1(j1), peak_range)
            flag1='True';
            break
        end
    end
    
    flag2='False';
    for j2=1:length(peaks2)
        if ismember(peaks2(j2), peak_range)
            flag2='True';
            break
        end
    end
    
    flag3='False';
    for j3=1:length(peaks3)
        if ismember(peaks3(j3), peak_range)
            flag3='True';
            break
        end
    end
    
%    if (strcmp(flag1, 'True') && strcmp(flag2, 'True') && strcmp(flag3, 'True'))
    if sum([strcmp(flag1, 'True'), strcmp(flag2, 'True'), strcmp(flag3, 'True')]) >= 2 
        %%write flags to file
        flags=[flag1, flag2, flag3];
        f_flags=fopen('flags_comb.txt', 'a');
        fwrite(f_flags,flags);
        fwrite(f_flags, '\n');
        fclose(f_flags);
        %%
        peak_out = [peak_out peak_in(i)];
        peaks1(j1)=[];
        peaks2(j2)=[];
        peaks3(j3)=[];
    end
end

if isempty(peak_out)
    peak_out = 0;
end