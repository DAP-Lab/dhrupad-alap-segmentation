import sys
import os
import librosa
import pandas as pd
import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from params import *

##Zero pad ends of spectrogram
def zeropad2d(X,n_frames):
	Y=np.hstack((np.zeros([X.shape[0],n_frames]), X))
	Y=np.hstack((Y,np.zeros([X.shape[0],n_frames])))
	return Y

##Compute mel spectrogram
def get_mel_specgram(audio, sr, win_len, hop_len, nfft, n_mels):
	mel_specgram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=nfft, hop_length=int(sr*hop_len), win_length=int(sr*win_len), window='hann', center=True, pad_mode='reflect', power=1.0, n_mels=n_mels, fmin=80)
	mel_specgram=10*np.log10(1e-5+mel_specgram)
	return mel_specgram

##Average the mel-spectrogram over texture windows at a larger hop
def overlap_resample(data, texwin_len, sub_factor):
	data_resampled=np.zeros([data.shape[0], int(data.shape[1]/sub_factor)])
	for i_frame in range(data_resampled.shape[1]):
		data_resampled[:,i_frame]=np.mean(data[:,i_frame*sub_factor:(i_frame*sub_factor)+texwin_len],1)
	return data_resampled

def get_frame_level_melgrams(audio):
	mel_specgram=get_mel_specgram(audio, sr, win_len, hop_len, nfft, n_mels)
	mel_specgram=overlap_resample(mel_specgram, int(texwin_len/hop_len), subFactor)
	mel_specgram_frames=np.zeros([mel_specgram.shape[1], mel_specgram.shape[0], 2*context_len,1])
	mel_specgram=zeropad2d(mel_specgram,context_len)
	
	for i_frame in range(mel_specgram_frames.shape[0]):
		mel_specgram_frame=np.array(mel_specgram[:,i_frame:i_frame+(2*context_len)], dtype='float32')

		#pad column of zeros if length lesser than context length
		if mel_specgram_frame.shape[1]<2*context_len:
			mel_specgram_frame=np.hstack((mel_specgram_frame, np.zeros([mel_specgram_frame.shape[0],2*context_len - mel_specgram_frame.shape[1]])))

		#range normalisation
		mel_specgram_frame=mel_specgram_frame/np.max(np.abs(mel_specgram_frame))
		mel_specgram_frames[i_frame,:,:,0]=mel_specgram_frame
	
	return mel_specgram_frames

def get_test_data(audio_filepath='', gt_boundaries=[], audio_offset=[0]):
	smear_win=windows.get_window('boxcar', targ_smear_width)

	data_test={'features':np.array([]), 'labels':np.array([])}
	for offset in audio_offset:
		audio, sr = librosa.load(audio_filepath, sr=16000)
		audio=audio[int(sr*offset):]
		
		features = get_frame_level_melgrams(audio)
		audio=[]

		labels=np.zeros(features.shape[0])
		for boundary in gt_boundaries:
			boundary=int(boundary.strip()) - offset
			labels[int(boundary/hop_len_sub)-targ_smear_width//2:int(boundary/hop_len_sub)+targ_smear_width//2]=smear_win
		
		if len(data_test['features'])==0: data_test['features'] = features
		else: data_test['features']=np.append(data_test['features'],features,axis=0)
		data_test['labels']=np.append(data_test['labels'], labels)

	return data_test

def build_model(input_dim):
	model = Sequential()

	model.add(Convolution2D(filters = 16, kernel_size = (3, 6), activation = 'relu', padding='same', input_shape=input_dim, kernel_initializer=keras.initializers.glorot_uniform(seed=0)))
	model.add(MaxPooling2D(pool_size = (3,3)))
	model.add(Convolution2D(filters = 32, kernel_size = (3, 6), activation = 'relu', padding='same',kernel_initializer=keras.initializers.glorot_uniform(seed=0)))
	model.add(MaxPooling2D(pool_size = (3,3)))
	
	model.add(Flatten())
	model.add(Dense(units = 128, activation = 'sigmoid',kernel_initializer=keras.initializers.glorot_uniform(seed=0)))
	model.add(Dropout(0.5,seed=0))
	model.add(Dense(units = 2, activation = 'softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=0)))
	return model

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

def eval_output(out_labels, out_probs, ground_truth, eval_tol, context_len, plot_savepath, plot_name, plot=True):
	diffGT=np.abs(ground_truth-np.append(0,ground_truth[:-1]))
	positives=np.where(diffGT==1.0)[0]
	ground_truth=np.zeros(len(ground_truth))
	for i_pos in range(0,len(positives),2):
		ground_truth[positives[i_pos]+ int((positives[i_pos+1]-positives[i_pos])/2)]=1
	peaksGt=ground_truth
	peakLocsGt=np.where(ground_truth==1.0)[0]

	peaksOut=get_peaks(out_labels, strengths=out_probs)
	peaksOut=merge_onsets(peaksOut,out_probs,eval_tol)
	
	nPositives=len(np.where(peaksGt==1.0)[0])
	nTP=0; nFP=0;
	
	if plot==True: plt.plot(peaksGt,'b',label='GT'); plt.plot(-peaksOut, 'r', label='Pred.'); plt.plot(out_probs,'g',label='Pred. prob.'); plt.legend(); plt.savefig(os.path.join(plot_savepath, plot_name+'.png')); plt.clf()

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
