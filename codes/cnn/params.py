import os

texwin_len=3
context_len=10 #+-context_len seconds

sr=16000
win_len=30e-3
hop_len=10e-3
nfft=1024
n_mels=40
eval_tol=30 		#+-eval_tol/2 seconds
targ_smear_width=30 #+- targ_smear_width/2 seconds

subFactor=100
hop_len_sub=hop_len*subFactor
eval_tol=int(eval_tol/hop_len_sub)
context_len=int(context_len/hop_len_sub)
targ_smear_width=int(targ_smear_width/hop_len_sub)

#Directories
songdata_filepath='../../annotations/GT_boundaries_sections.csv'
testsongdata_filepath='../../annotations/GT_boundaries_sections_test.csv'
audio_dir='/media/Sharedata/rohit/DrupadAlapaudios'

features_out_path='../../features/feats_labels_CNN/texwin_%d_context_%d/'%(texwin_len, context_len)
model_savepath = '../../saved_models/CNN/%d_%d'%(texwin_len,context_len)
plot_savepath = '../../plots/CNN/%d_%d'%(texwin_len,context_len)
log_savepath = '../../logs/CNN'

if not os.path.exists(model_savepath): os.makedirs(model_savepath)
if not os.path.exists(log_savepath): os.makedirs(log_savepath)
if not os.path.exists(plot_savepath): os.makedirs(plot_savepath)

#k-fold cross-val
n_folds=20
