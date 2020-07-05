% Function to give self similarity matrix
% Input   : Kw-> floor(Kernel Width/2)
%         : dist_measure->Distance measure for SDM computation
%          : As per the matlab version and the distances allowed
%         : feature-> Feature matrix

% Output : Similarity matrix sim_mat
%        : Novelty score 

%  Example Usage: 
% [sim_mat,nov_score]=SDM_nov(50,'Euclidean',feature);

function [sim_mat,nov_score]= sdm_nov(Kw,dist_measure,feature)

%%%  make a kernel of 2*Kw+1

zv=zeros(Kw,1);

zh=zeros(1,2*Kw+1);

pos_mat=ones(Kw,Kw);
neg_mat=-1*ones(Kw,Kw);


kernel=horzcat(neg_mat,zv,pos_mat);
kernel=vertcat(kernel,zh);

kernel1=horzcat(pos_mat,zv,neg_mat);
kernel=vertcat(kernel,kernel1);



% compute a similarity matrix
sim_mat=pdist2(feature,feature,dist_measure);



nov_score=zeros(length(sim_mat),1);


%  kernel multiplication along diagonal to obtain the novelty score

for i=Kw+1:length(sim_mat)-Kw-1
       
    for k1=-Kw:Kw
        for k2=-Kw:Kw
    temp_chk(k1+Kw+1,k2+Kw+1)= sim_mat(i+k1,i+k2);
        end
    end
    nov_score(i)=sum(sum(temp_chk.*kernel));
end
