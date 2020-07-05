function [hrate, falarms] = hratefalarms(nov_peaks_detected, gt, threshold)
time = nov_peaks_detected;
c = 1;

gt = gt(gt~=0);
for i=1:length(time)
    temp(i,:) = time(i)-threshold/2:0.1:time(i)+threshold/2;
end
hrate =0;
for i=1:length(gt)
    if ismember(gt(i),temp)
        hrate = hrate+1;
    end
end
falarms = length(time)-hrate;
hrate = hrate;
%hrate = hrate/length(gt);