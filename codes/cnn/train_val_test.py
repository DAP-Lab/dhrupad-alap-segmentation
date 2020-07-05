import sys
import os
import fnmatch
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import windows
from sklearn.utils import shuffle
import keras
import tensorflow as tf
import utils
from params import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(0)
train_flag=int(sys.argv[1])

def cross_val(fold):
	#build model
	model_loss='binary_crossentropy'
	input_dim=(n_mels,2*context_len,1)
	clf=[]
	clf=utils.build_model(input_dim)
	clf.compile(optimizer = 'adam', loss = model_loss, metrics = ['accuracy'])
	if fold==0: print(clf.summary())

	#test song indices
	song_ids_val=song_ids[fold*fold_size:(fold+1)*fold_size]
	
	#split filenames into train and val
	partition={'train':np.array([]), 'val':np.array([])}
	for song_id in np.arange(songdata.shape[0]):
		if song_id not in song_ids_val:
			partition['train']=np.append(partition['train'], dataset_ids[str(song_id)])				

	#load data files
	data_train={'features': np.zeros([len(partition['train']), input_dim[0], input_dim[1], input_dim[2]]), 'labels': []}
	for i_samp in range(len(partition['train'])):
		data_train['features'][i_samp,]=np.load(os.path.join(datadir, partition['train'][i_samp] + '.npy'), allow_pickle=True)
		data_train['labels'].append(labels_all[partition['train'][i_samp]])
	
	#shuffle data
	data_train['features'], data_train['labels'] = shuffle(data_train['features'], data_train['labels'], random_state=0)

	#make labels one-hot
	data_train['labels'] = keras.utils.to_categorical(data_train['labels'], num_classes=2)

	print('===\nCross-validation fold no. %d\n===\n'%song_ids_val[0])

	#iteratively calling model fit function over batches and epochs
	batch_size=64
	n_batches=int(data_train['features'].shape[0]/batch_size)
	n_epochs=20
	
	train_losses=[]; eval_losses=[1e5]
	for i_epoch in range(n_epochs):
		#train
		train_loss=0
		for i_batch in range(n_batches):
			data_batch = data_train['features'][i_batch*batch_size:(i_batch+1)*batch_size]
			labels_batch = data_train['labels'][i_batch*batch_size:(i_batch+1)*batch_size]
			train_loss_batch,_ = clf.train_on_batch(data_batch, labels_batch)
			train_loss += train_loss_batch
		train_losses.append(train_loss/n_batches)
		
		#evaluate to save best model across epochs
		eval_loss=0
		for song_id in song_ids_val:
			boundaries=songdata['Boundaries'][song_id].split(',')
			filepath=os.path.join(audio_dir, songdata['Concert name'][song_id]+'.wav')
			data_val=utils.get_test_data(audio_filepath=filepath, samp_rate=sr, gt_boundaries=boundaries, offset=0)
			data_val['labels']=keras.utils.to_categorical(data_val['labels'], num_classes=2)

			eval_loss_song,_ = clf.evaluate(data_val['features'],data_val['labels'],verbose=0)
			eval_loss+=(eval_loss_song/len(song_ids_val))
			
		if eval_loss<np.min(eval_losses):
			clf.save_weights(os.path.join(model_savepath,'weights_fold%d.hdf5'%fold))
		eval_losses.append(eval_loss)
			
		print('\nEpoch %d/%d\tTrain Loss: %1.3f\tVal Loss: %1.3f'%(i_epoch+1, n_epochs, train_losses[-1], eval_losses[-1]))


	#Validate saved model on left-out song
	clf.load_weights(os.path.join(model_savepath,'weights_fold%d.hdf5'%fold))

	scores=[]
	for song_id in song_ids_val:
		scores_song=[]
		for offset in audio_offset_list:
			scores_offset=[]
			boundaries=songdata['Boundaries'][song_id].split(',')
			filepath=os.path.join(audio_dir, songdata['Concert name'][song_id]+'.wav')
			data_val = utils.get_test_data(audio_filepath=filepath, samp_rate=sr, gt_boundaries=boundaries, offset=offset)
			out_probs = clf.predict(data_val['features'])[:,1]

			for predict_thresh in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
				out_labels = (out_probs > predict_thresh).astype(int)
				plot_name = 'fold%d_'%fold + songdata['Concert name'][song_id] + '_thresh_' + str(predict_thresh)
				scores_offset = np.append(scores_offset, utils.eval_output(out_labels, out_probs, data_val['labels'], eval_tol, context_len, plot_savepath, plot_name))

			if len(scores_song)==0: scores_song=scores_offset
			else: scores_song+=scores_offset

		if len(scores)==0:
			scores=np.atleast_2d(scores_song)
		else:
			scores=np.vstack((scores,np.atleast_2d(scores_song)))
	return scores

if __name__=='__main__':
	datadir=features_out_path

	#load song names and ground truth boundaries
	songdata=pd.read_csv(songdata_filepath)
	song_ids=np.arange(songdata.shape[0])
	songdata_test=pd.read_csv(testsongdata_filepath)
	
	fold_size=int(songdata.shape[0]/n_folds)
	
	#load data files
	dataset_ids=np.load(os.path.join(datadir, 'dataset_ids.npy'), allow_pickle=True).item()
	labels=np.load(os.path.join(datadir, 'labels.npy'), allow_pickle=True).item()
		
	#flatten song-wise labels dict above to one dict containing label for every frame of every song
	labels_all={}
	for label_song in labels:
		for label_item in labels[label_song]:
			labels_all[label_item]=labels[label_song][label_item]

	flog=open(os.path.join(log_savepath,'CNN_train_test.txt'),'a')
	flog.write('----------texwin_len=%d\tcontext_len=%d\t-----------\n'%(texwin_len, context_len))

	scores=[]
	#train & validate model, and report Precision, Recall & F-score
	if train_flag:
		for i_fold in [19]: #range(n_folds):
			if len(scores)==0:
				scores=np.atleast_2d(cross_val(i_fold))
			else:
				scores=np.vstack((scores,np.atleast_2d(cross_val(i_fold))))

			for score in scores[-1,:]:
				flog.write('%d\t'%score)
			flog.write('\n')

		##calculate and write Precision, Recall and F-score to log file
		scores=np.sum(scores,0)
		flog.write('Precision:\t')
		for i_thresh in range(0,len(scores),3):
			flog.write('%f\t'%(scores[i_thresh]/(scores[i_thresh]+scores[i_thresh+1])))
		flog.write('\n')
		
		flog.write('Recall:\t')
		for i_thresh in range(0,len(scores),3):
			flog.write('%f\t'%(scores[i_thresh]/scores[i_thresh+2]))
		flog.write('\n')
		
		flog.write('F-score:\t')
		for i_thresh in range(0,len(scores),3):
			flog.write('%f\t'%(2*(scores[i_thresh]/(scores[i_thresh]+scores[i_thresh+1]))*(scores[i_thresh]/scores[i_thresh+2])/((scores[i_thresh]/(scores[i_thresh]+scores[i_thresh+1]))+(scores[i_thresh]/scores[i_thresh+2]))))
		flog.write('\n=====================\n\n')

	#test saved model on test data
	else:
		##Test
		flog.write('--Test scores--\n')

		#load weights from disk
		test_model_path = os.path.join(model_savepath, 'weights_fold0.hdf5')

		input_dim=(n_mels,2*context_len,1)
		clf=utils.build_model(input_dim)
		clf.load_weights(test_model_path)
		
		for song_id in range(songdata_test.shape[0]):
			flog.write(songdata_test['Concert name'][song_id]+'\n')
			filepath = os.path.join(audio_dir, songdata_test['Concert name'][song_id]+'.wav')
			try:	boundaries=songdata_test['Boundaries'][song_id].split(',')
			except:	boundaries=[]

			for offset in audio_offset_list:
				data_test=utils.get_test_data(audio_filepath=filepath, samp_rate=sr, gt_boundaries=boundaries, audio_offset=[offset])
				out_probs=clf.predict(data_test['features'])[:,1]
				out_labels=(out_probs > 0.6).astype(int)
					
				if len(boundaries)==0:
					plt.plot(out_probs); plt.plot(out_labels); plt.show()
				
				#if boundaries available, evaluate predictions
				else:			
					plot_name=songdata_test['Concert name'][song_id]+'_'+str(offset)
					scores_offset=[utils.eval_output(out_labels, out_probs, data_test['labels'], eval_tol, context_len, plot_savepath, plot_name)]		
					if len(scores)==0:
						scores=np.atleast_2d(scores_offset)
					else:
						scores=np.vstack((scores,np.atleast_2d(scores_offset)))
					for score in scores[-1,:]:
						flog.write('%d\t'%score)
					flog.write('\n')					
					
			if len(boundaries)!=0:
				##Calculate and write Precision, Recall and F-score to log file
				scores=np.sum(scores,0)
				flog.write('Precision:\t')
				flog.write('%f\n'%(scores[0]/(scores[0]+scores[1])))
				flog.write('Recall:\t')
				flog.write('%f\n'%(scores[0]/scores[2]))
				flog.write('F-score:\t')
				flog.write('%f\n'%(2*(scores[0]/(scores[0]+scores[1]))*(scores[0]/scores[2])/((scores[0]/(scores[0]+scores[1]))+(scores[0]/scores[2]))))
				flog.write('\n=====================\n\n')
				
flog.close()
