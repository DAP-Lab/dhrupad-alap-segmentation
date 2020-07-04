import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import windows
import scipy.io as sio

def make_feature_subsets(data, context_len):
	""" Make 'rhythm', 'mfcc', 'timbre' and 'all' subsets from features (refer to Table 2 of paper), and add context to every frame.
	    Context is added by appending to the feature vector of every frame, the feature vector of the past and future frames.

	Parameters:
	--
	data (dict): dictionary of features from the structure saved in the mat file
	context_len (int): number of frames of context (for each of previous and future frames)

	Returns:
	--
	dict: dictionary of the 4 feature subsets
	"""
	data_subsets={}
	#data_subsets['max_P']=np.atleast_2d(np.max(data['P'],1)).T
	data_subsets['max_P']=np.atleast_2d(np.max(data['rhythm_features'],1)).T
	data_subsets['rhythm']=data_subsets['max_P']
	data_subsets['rhythm']=get_context_features(data_subsets['rhythm'], context_len)
	data_subsets['mfcc']=data['avg_MFCC']
	data_subsets['mfcc']=get_context_features(data_subsets['mfcc'], context_len)
	#data_subsets['timbre']=np.hstack((data['scaled_features'][:,3:5], data['avg_MFCC']))
	data_subsets['timbre']=np.hstack((data['ste_sc_diff'], data['avg_MFCC']))
	data_subsets['timbre']=get_context_features(data_subsets['timbre'], context_len)
	#data_subsets['all']=np.hstack((data_subsets['max_P'],data['scaled_features'][:,3:5], data['avg_MFCC']))
	data_subsets['all']=np.hstack((data_subsets['max_P'],data['ste_sc_diff'], data['avg_MFCC']))
	data_subsets['all']=get_context_features(data_subsets['all'], context_len)		
	return	data_subsets

def get_context_features(dataset, context_len):
	""" Add context in the form of appending the features from a few previous and next frames, to every frame's feature vector
	
	Parameters:
	--
	dataset (numpy ndarray): data matrix containing the features
	context_len (int): duration of context in number of frames
	
	Returns:
	--
	numpy ndarray
	"""
	n_feats=dataset.shape[1]

	#first, the past frames
	for i_len in range(context_len):
		dataset=np.hstack((np.vstack((np.zeros([i_len+1,n_feats]),dataset[:-(i_len+1),:n_feats])),dataset))
	
	#then, the future frames
	for i_len in range(context_len):
		dataset=np.hstack((dataset,np.vstack((dataset[i_len+1:,:n_feats],np.zeros([i_len+1,n_feats])))))
	
	return dataset

def get_offset_features(features_matlab_path, songName, data_out, target_smear_width, frame_len, context_len, audio_offset, pitch_shift, boundaries):
	"""Load feature data files of offset hop durations and append only the boundary-neighborhood frames to the dataset. To improve data class imbalance.
	
	Parameters:
	--
	songName (string): name of the data file
	data_out (dict): features for default hop duration
	target_smear_width (int): size of boundary-neighborhood (in seconds)
	frame_len (float): duration of a frame (in seconds)
	context_len (int): number of +-context frames
	audio_offset (float): offset value (in seconds) added to audio signal
	pitch_shift (int): pitch shift (in +/-semitones)
	boundaries (numpy array or list): set of boundary positions for the song (in seconds)
	
	Returns:
	--
	numpy ndarray: feature data for frames in the neighborhood
	"""
	
	data_offset=sio.loadmat(os.path.join(features_matlab_path,songName+'_'+str(audio_offset)+'_'+str(pitch_shift)))
	data_offset=make_feature_subsets(data_offset, context_len)
	
	for bound in boundaries:
		bound=int(bound.strip()) - audio_offset
		bound_frame=int(bound/frame_len)
		data_out['rhythm']=np.vstack((data_out['rhythm'],data_offset['rhythm'][bound_frame-int(target_smear_width/2):bound_frame+int(target_smear_width/2),:]))
		data_out['mfcc']=np.vstack((data_out['mfcc'],data_offset['mfcc'][bound_frame-int(target_smear_width/2):bound_frame+int(target_smear_width/2),:]))
		data_out['timbre']=np.vstack((data_out['timbre'],data_offset['timbre'][bound_frame-int(target_smear_width/2):bound_frame+int(target_smear_width/2),:]))
		data_out['all']=np.vstack((data_out['all'],data_offset['all'][bound_frame-int(target_smear_width/2):bound_frame+int(target_smear_width/2),:]))
		data_out['labels']=np.vstack((data_out['labels'], np.atleast_2d(windows.get_window('boxcar',target_smear_width)).T))

	return data_out

def load_dataset(songdata,feat_subset,feat_path):
	dataset={'features':np.array([]), 'labels':np.array([])}
	song_lengths=np.array([0])
	
	for i_song in range(songdata.shape[0]):
		feat_label_data=np.load(os.path.join(feat_path,songdata['Concert name'][i_song]+'.npy'), allow_pickle=True).item()
		if len(dataset['features'])==0:
			dataset['features']=feat_label_data[feat_subset]
			dataset['labels']=feat_label_data['labels']
		else:
			dataset['features']=np.vstack((dataset['features'],feat_label_data[feat_subset]))
			dataset['labels']=np.vstack((dataset['labels'],feat_label_data['labels']))
		song_lengths=np.append(song_lengths,dataset['features'].shape[0])
	
	return dataset, song_lengths

def get_train_set(dataset, song_lengths, song_inds_test):
	train_set={'features':np.array([]), 'labels':np.array([])}

	for i_song in range(len(song_lengths)-1):
		if i_song not in song_inds_test:
			if len(train_set['features'])==0:
				train_set['features']=dataset['features'][song_lengths[i_song]:song_lengths[i_song+1]]
				train_set['labels']=dataset['labels'][song_lengths[i_song]:song_lengths[i_song+1]]
			else:
				train_set['features']=np.vstack((train_set['features'],dataset['features'][song_lengths[i_song]:song_lengths[i_song+1]]))
				train_set['labels']=np.vstack((train_set['labels'],dataset['labels'][song_lengths[i_song]:song_lengths[i_song+1]]))

	return train_set

def get_val_set(context_len, feat_subset, feat_path, song_num=None, songdata=[], data_filepath='', boundaries=[], offset=0):
	val_set={'features':np.array([]), 'labels':np.array([])}

	if song_num!=None:
		data_filepath=os.path.join(feat_path,songdata['Concert name'][song_num])
		boundaries=songdata['Boundaries'][song_num].split(',')
		data=sio.loadmat(data_filepath + '_%s'%(str(offset)))

	else:
		data=sio.loadmat(data_filepath.replace('.mat',''))

	data=make_feature_subsets(data,context_len)

	if boundaries==[]:
		if len(val_set['features'])==0:
			val_set['features']=data[feat_subset]
		else:
			val_set['features']=np.vstack((val_set['features'],data[feat_subset]))

	else:
		labels=np.zeros(data['all'].shape[0])
		for boundary in boundaries:
			boundary=int(int(boundary.strip()) - offset)
			labels[boundary]=1
		labels=np.atleast_2d(labels).T
		
		if len(val_set['features'])==0:
			val_set['features']=data[feat_subset]
			val_set['labels']=labels
		else:
			val_set['features']=np.vstack((val_set['features'],data[feat_subset]))
			val_set['labels']=np.vstack((val_set['labels'],labels))
	return val_set
	
def get_peaks(x, strengths=[]):
	i=0
	while i < len(x):
		if x[i]==1.0:
			i_start=i
			flag=True
			while flag==True:
				i+=1
				if i>=len(x):
					flag=False
					i_end=i
					x[i_start:i_end]=0
					if len(strengths)==0: x[i_start + int((i_end-i_start)/2)]=1
					else: x[i_start + np.argmax(strengths[i_start:i_end])]=1
				elif x[i]==0:
					flag=False
					i_end=i
					x[i_start:i_end]=0
					if len(strengths)==0: x[i_start + int((i_end-i_start)/2)]=1
					else: x[i_start + np.argmax(strengths[i_start:i_end])]=1			
		else: i+=1
	return x

def merge_onsets(onsets,strengths,mergeDur):
	onsetLocs=np.where(onsets==1)[0]
	ind=1
	while ind<len(onsetLocs):
		if onsetLocs[ind]-onsetLocs[ind-1] < mergeDur:
			if strengths[onsetLocs[ind]]<strengths[onsetLocs[ind-1]]:
				onsets[onsetLocs[ind]]=0
				onsetLocs=np.delete(onsetLocs,ind)
			else:
				onsets[onsetLocs[ind-1]]=0
				onsetLocs=np.delete(onsetLocs,ind-1)
		else: ind+=1
	return onsets

def smooth_predictions(outLabels, outProbs, merge_dur):
	peaksOut=get_peaks(outLabels, strengths=outProbs)
	peaksOut=merge_onsets(peaksOut,outProbs,merge_dur)
	return peaksOut

def eval_output(outLabels, outProbs, groundTruth, eval_tol, plot_savepath, plot_name):
	peaksOut=smooth_predictions(outLabels,outProbs,eval_tol)
	peaksGt=groundTruth
	peakLocsGt=np.where(peaksGt==1)[0]

	plt.plot(peaksGt,'b', label='GT'); plt.plot(-peaksOut, 'r', label='Pred.'); plt.plot(outProbs,'g', label='Pred. prob.'); plt.title(str(plot_name)); plt.xlabel('Time(s)', fontsize=14); plt.legend(fontsize=12); plt.savefig(os.path.join(plot_savepath, plot_name)+'.png'); plt.clf()

	nPositives=len(peakLocsGt)
	nTP=0; nFP=0;

	i_peak=0
	while i_peak < len(peaksOut):
		if peaksOut[i_peak]!=0.:
			if len(peakLocsGt)==0:
				nFP+=1
				i_peak+=1
			elif abs(i_peak-peakLocsGt[0])<=int(eval_tol/2):
				nTP+=1
				peakLocsGt=np.delete(peakLocsGt,0)
				i_peak+=1
			elif i_peak<peakLocsGt[0]:
				nFP+=1
				i_peak+=1
			elif i_peak>peakLocsGt[0]:
				peakLocsGt=np.delete(peakLocsGt,0)
		else: i_peak+=1

	return nTP, nFP, nPositives


