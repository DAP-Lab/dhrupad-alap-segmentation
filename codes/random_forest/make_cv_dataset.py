import sys
import os
import scipy.io as sio
import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt
import pandas as pd
import utils
from params import *

def save_feats_labels(songdata,audio_offsets=[],pitch_shifts=[]):
	""" Generate frame-level labels from GT boundary annotations (binary: 1 for boundary, 0 for non-boundary) and save the features and labels together
	
	Parameters:
	--
	_songdata (pd DataFrame): data from GT annotations csv file
	"""

	for songNo in range(songdata.shape[0]):
		boundaries=songdata['Boundaries'][songNo].split(',')

		data=sio.loadmat(os.path.join(features_matlab_path,songdata['Concert name'][songNo]))		
		data_out=utils.make_feature_subsets(data,context_len)		

		n_frames=data_out['all'].shape[0]

		smear_win=windows.get_window('boxcar',target_smear_width)

		boundLabels=np.zeros(n_frames)
		for i_bound in range(1, len(boundaries)):
			boundary=int(boundaries[i_bound].strip())
			boundLabels[boundary//frame_len - target_smear_width//2:boundary//frame_len + target_smear_width//2]=smear_win
		data_out['labels']=np.atleast_2d(boundLabels).T

		for pitch_shift in pitch_shifts:
			for audio_offset in audio_offsets:
				if (pitch_shift==0) & (audio_offset==0): continue
				data_out=utils.get_offset_features(features_matlab_path, songdata['Concert name'][songNo], data_out, target_smear_width, frame_len, context_len, audio_offset, pitch_shift, boundaries)

		np.save(os.path.join(features_RF_path,songdata['Concert name'][songNo]+'.npy'),data_out)
	return

if __name__=='__main__':
	songdata=pd.read_csv(songdata_filepath)
	save_feats_labels(songdata,audio_offset_list,pitch_shift_list)
