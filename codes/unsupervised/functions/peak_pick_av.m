% Peak picking algorithm %  
% Input : nov_fn -> Novelty function
%       : md ->  Median filter length 
%       : me -> Mean filter length  
%       : Thresh_search -> Vicinity to search for the peak in the original
%                          nov_fn from the detected peaks in smootened nov_fn
%       :  thres -> Percentage w.r.t highest peak to consider as  valid peak (say 0.1) 

% Output: Valid location of the peak
%       : Height of the peak

function [peaks_detected,me_nov,med_nov]=peak_pick_av(nov_fn,md,me,th_search,thres)

med_nov=medfilt1(nov_fn,md);     % Taken as 40
me_nov=movavg(med_nov,me,me);    % Taken as 30

% Detect peaks in the original contour
ct=0;
for i=3:length(nov_fn)-2
    if(nov_fn(i)>nov_fn(i-1) && nov_fn(i)>nov_fn(i-2) && nov_fn(i)>=nov_fn(i+1) && nov_fn(i)>=nov_fn(i+2))    %5 point peak picking
        ct=ct+1;
        orig_peak(ct,1)=i;
        orig_peak(ct,2)=nov_fn(i);
    end
end
orig_peak_locn=zeros(1,length(nov_fn));
orig_peak_locn(orig_peak(:,1))=nov_fn(orig_peak(:,1));

% Detect peaks in the smoothened contour
ct=0;
for i=3:length(nov_fn)-2
    if(me_nov(i)>me_nov(i-1) && me_nov(i)>me_nov(i-2) && me_nov(i)>=me_nov(i+1) && me_nov(i)>=me_nov(i+2))
        ct=ct+1;
        coarse_peak(ct,1)=i;
        coarse_peak(ct,2)=me_nov(i);
    end
end


% Detect peaks in the original contour from the ones found in the smoothed
% contour
fct=0;
for k=1:ct
    
    temp_loc=coarse_peak(k,1);
    if(temp_loc+th_search<length(orig_peak_locn))
    temp_nov=orig_peak_locn(abs(temp_loc-th_search):temp_loc+th_search);
    
    [max_val max_loc]=max(temp_nov);
    
    if(max_val>=thres*max(orig_peak_locn))
    fct=fct+1;
    peaks_detected_init(fct,1)=temp_loc+max_loc-th_search-1;
    
    peaks_detected_init(fct,2)=max_val;
    end
    end
end
peaks_new=unique(peaks_detected_init,'rows');
no_zr_pk=peaks_new(:,1)~=0;
peaks_detected=peaks_new(no_zr_pk==1,:);

    
    
        
    





