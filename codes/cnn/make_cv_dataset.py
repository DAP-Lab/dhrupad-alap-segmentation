import sys
import os
import librosa
import pandas as pd
import numpy as np
from scipy.signal import windows
import utils
from params import *

np.random.seed(0)
	
##Save frame-wise context_len long chunks of the mel-spectrogram and corresponding binary targets as training examples
def gen_dataset(features_out_path, texwin_len, context_len, targ_smear_width, audio_offsets=[0], pitch_shifts=[0]):
	
	smear_win=windows.get_window('boxcar',targ_smear_width)

	dataset_ids={}
	dataset_labels={}
	print('\n----------\n')
	for i_song in range(songdata.shape[0]):
			print('Generating data for song no %d'%i_song)
			dataset_ids[str(i_song)]=[]			
			dataset_labels[str(i_song)]={}
			boundaries=songdata['Boundaries'][i_song].split(',')

			for i_pitch in pitch_shifts:
				if i_pitch==0:
					filepath=os.path.join(audio_dir, songdata['Concert name'][i_song] + '.wav')
				else:
					filepath=os.path.join(audio_dir, 'pitch_shifted', songdata['Concert name'][i_song] + '_' + str(i_pitch) + '.wav')

				for offset in audio_offsets:
					audio, sr = librosa.load(filepath, sr=16000)
					audio=audio[int(sr*offset):]
					
					features = utils.get_frame_level_melgrams(audio)
					audio=[]
					
					labels=np.zeros(features.shape[0])
					for boundary in boundaries:
						boundary=int(boundary.strip()) - offset
						labels[int(boundary/hop_len_sub)-targ_smear_width//2:int(boundary/hop_len_sub)+targ_smear_width//2]=smear_win

					dataset_ids_temp=[]	#IDs of this version of the audio
					labels_temp=[]		#retaining labels only in boundary neighborhood
					
					for i_frame in range(features.shape[0]):
						ID='%d_%d_%f_%d'%(i_song, i_pitch, offset, i_frame)
						mel_specgram_frame=features[i_frame,]

						if(((i_pitch==0) & (offset==0)) | (labels[i_frame]>0)):
							np.save(os.path.join(features_out_path,ID), np.reshape(mel_specgram_frame, (mel_specgram_frame.shape[0], mel_specgram_frame.shape[1], 1)))
							dataset_ids_temp.append(ID)
							labels_temp.append(labels[i_frame])

					dataset_ids[str(i_song)].extend(dataset_ids_temp)
					dataset_labels[str(i_song)].update(dict(zip(dataset_ids_temp, labels_temp)))
	return dataset_ids, dataset_labels


if __name__=='__main__':
	if os.path.exists(features_out_path): os.system('rm -rf %s'%features_out_path)
	os.makedirs(features_out_path)

	songdata=pd.read_csv(songdata_filepath)

	dataset_ids, labels = gen_dataset(features_out_path, texwin_len, context_len, targ_smear_width,audio_offset_list,pitch_shift_list)
	np.save(os.path.join(features_out_path, 'dataset_ids'), dataset_ids)
	np.save(os.path.join(features_out_path, 'labels'), labels)
