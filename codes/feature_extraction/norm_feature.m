% %%%% Feature Normalization
% 
% Input : feature array matrix( time stamps vs features)
%       : Char input as 'a', 'b', 'c'
%        Scaling a as normalization to standard gaussain
%        Scaling b as max normalization 
%        Scaling c as Min max normalization
%  
% Output : Scaled features according to one of the options

function feature_scale=norm_feature(feature,Scale_par)

sz=size(feature);

if(Scale_par=='a')
for i=1:sz(2)
    temp=feature(:,i);
    temp1=(temp-mean(temp))/std(temp);
    
    feature_scale(:,i)=temp1;
end

elseif(Scale_par=='b')
    for i=1:sz(2)
    temp=feature(:,i);
    temp1=temp/max(temp);
    
    feature_scale(:,i)=temp1;
    end

elseif(Scale_par=='c')
    for i=1:sz(2)
    temp=feature(:,i);
    temp1=(temp-min(temp))/(max(temp)-min(temp));
    
    feature_scale(:,i)=temp1;
    end
else
    feature_scale=feature;
    
end  