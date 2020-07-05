function [peak, peak_val] = find_peak_2(signal,n,nbr)
%this algorithm obtains deepest n troughs and if their are more than 1 of 
%troughs in threshold of '+/-nbr' then it takes only the deepest trough
%among them
check = 2;
c=1;
for i = 1+check:length(signal)-check
    lgc = 1;
    for j=1:check
        lgc = (signal(i-j)>signal(i))*lgc*(signal(i+j)>signal(i));
    end
    if lgc 
        peak_candidate(c) = i;
        val(c) = signal(i);
        c = c+1;
    end
end
 if (length(peak_candidate)< n)
     n= length(peak_candidate);
 end    

[B,I] = sort(val);
peak(1) = peak_candidate(I(1));
c = 1;
d = 2;
while(1)
    lgc = 1;
    for k = 1:c
        lgc = lgc * abs(peak(k)-peak_candidate(I(d)))>nbr/2;
    end
     if c>=n || d>=n
        break;
    end
    if lgc;
        peak(c+1) = peak_candidate(I(d));
        c = c+1;
    end
    d = d+1;
   
end
%peak = peak_candidate(I(1:n));
peak_val = signal(peak);