import os

#parameters
texwin_len=3
context_len=50 #+-context_len seconds

sr=16000
win_len=30e-3
hop_len=10e-3
nfft=1024
n_mels=40
eval_tol=30 		#+-eval_tol/2 seconds
targ_smear_width=30 #+- targ_smear_width/2 seconds
frame_len = 1

subFactor=100
hop_len_sub=hop_len*subFactor
eval_tol=int(eval_tol/hop_len_sub)
context_len=int(context_len/hop_len_sub)
targ_smear_width=int(targ_smear_width/hop_len_sub)

#parameters for data augmentation
audio_offset_list=[0,0.1,0.2,0.3,0.4]
pitch_shift_list=[0,1,2,3,4]

#Directories
songdata_filepath='../../annotations/train_dataset.csv'
testsongdata_filepath='../../annotations/test_dataset.csv'
audio_dir='/media/Sharedata/rohit/DrupadAlapaudios/filtered'

features_out_path='../../features/feats_labels_CNN/texwin_%d_context_%d/'%(texwin_len, context_len)
model_savepath = './saved_models/%d_%d'%(texwin_len,context_len)
plot_savepath = './plots/%d_%d'%(texwin_len,context_len)
log_savepath = './logs/'

if not os.path.exists(model_savepath): os.makedirs(model_savepath)
if not os.path.exists(log_savepath): os.makedirs(log_savepath)
if not os.path.exists(plot_savepath): os.makedirs(plot_savepath)

#k-fold cross-val
n_folds=20

#test model
test_model_path = os.path.join(model_savepath, 'weights_fold19.hdf5')

