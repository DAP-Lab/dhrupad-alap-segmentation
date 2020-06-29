import os
#parameters

feat_subset='all'		#one of: 'rhythm', 'mfcc', 'timbre', 'all'
texwin_len=3			#in seconds
context_len=5			#in seconds; adds context of +-context_len, i.e., 2*context_len
n_trees=5			#number of trees in the RF
eval_tol=30	 		#evaluation tolerance of +-eval_tol/2 seconds
target_smear_width=30		#in seconds

frame_len=1			#1s
context_len = int(context_len/frame_len)	#in frames
eval_tol = int(eval_tol/frame_len)		#in frames
target_smear_width = int(target_smear_width/frame_len)	#in frames

#song data paths
songdata_filepath='../../annotations/train_dataset.csv'
songdata_test_filepath='../../annotations/test_dataset.csv'

#input paths
features_matlab_path='/media/Sharedata/rohit/drupad-alap-segmentation/features/features_from_matlab/train_texwin%d'%texwin_len
testfeatures_matlab_path='/media/Sharedata/rohit/drupad-alap-segmentation/features/features_from_matlab/test_texwin%d'%texwin_len

#output paths
features_RF_path='../../features/feats_labels_RF/texwin_%d_context_%d'%(texwin_len, context_len)
model_savepath = '../../saved_models/RF/%s_%d_%d_%d'%(feat_subset,texwin_len,context_len,n_trees)
plot_savepath = '../../plots/RF/%s_%d_%d_%d'%(feat_subset,texwin_len,context_len,n_trees)
log_savepath = '../../logs/RF'

if not os.path.exists(features_RF_path): os.makedirs(features_RF_path)
if not os.path.exists(model_savepath): os.makedirs(model_savepath)
if not os.path.exists(plot_savepath): os.makedirs(plot_savepath)
if not os.path.exists(log_savepath): os.makedirs(log_savepath)

#saved model for testing
test_fold=-1	#train on all data if -1. Else, use the model from that CV fold
test_model_path = os.path.join(model_savepath, 'model_fold%d.sav'%test_fold)

