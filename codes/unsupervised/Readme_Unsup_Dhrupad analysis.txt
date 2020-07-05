1) Case study Alap analysis routine: DruAl_CaseStudyAl_analysis.m

 This routine picks up the song indicated by the user through song_ind, 
i)computes and displays the rhythmic features (tempo, salience), posterior feature, timbre features (ST energy, ST spectral centroid, MFCC) 
ii) Computes and displays the Self Distance matrix and the Novelty functions of from the features extracted 

2) Feature extraction routine: 


3) Boundary detection and evaluation routine: Unsup_BoundaryDetecn_Evaluation.m

This routine takec in the saved feature vectors of all songs, computes the SDM-Nov, does peak- picking and combines the peaks picked by rhythm and timbre. Finally, it evaluates Precison, Recall and F-scores. 