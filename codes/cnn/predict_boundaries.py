import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
import utils
from params import *

from datetime import datetime
now = datetime.now()

#cmd-line arguments
in_path=sys.argv[1]

if in_path[-3:] == 'wav':
	songs_list=[in_path]

elif in_path[-3:] in ['txt','csv']:
	try:
		songs_data=pd.read_csv(in_path)
		songs_list=songs_data['Concert name']
	except KeyError:
		songs_list=np.loadtxt(in_path)
	except:
		print('Invalid input. Accepted formats - filepath to a single wav file, a txt/csv file in the annotation file format, or txt/csv file containing a list of feature filenames')
		sys.exit()

flog=open(os.path.join(log_savepath,'CNN_test_log.txt'),'a')
flog.write(str(now)+'\n')

#load the model from disk
input_dim=(n_mels,2*context_len,1)
clf=utils.build_model(input_dim)
if not os.path.exists(test_model_path):
	print('Required model not available\n')
clf.load_weights(test_model_path)

#loop over test songs
for i_song in range(len(songs_list)):
	song_name=songs_list[i_song]
	flog.write(song_name+'\n')

	#if boundaries provided, read them
	try: boundaries=songs_data['Boundaries'][i_song].split(',')
	except: boundaries=[]

	scores=[]
	#loop over offset versions of test audio (test-time augmentation)
	for offset in audio_offset_list:
		if os.path.exists(song_name):
			filepath=song_name
		else:
			filepath = os.path.join(audio_dir, song_name)

		data_test=utils.get_test_data(audio_filepath=filepath, gt_boundaries=boundaries, offset=offset)
		out_probs=clf.predict(data_test['features'])[:,1]
		out_probs/=np.max(out_probs)
		out_labels=(out_probs > 0.3).astype(int)
		out_labels=utils.smooth_predictions(out_labels,out_probs,eval_tol)

		#if no boundaries available, plot output and save predicted boundaries to log
		if len(boundaries)==0:
			#plt.plot(out_probs); plt.plot(out_labels); plt.show()
			flog.write('Predicted boundaries:\t')
			for item in np.where(out_labels==1)[0]:
				flog.write('%5.2f s, '%(item*frame_len))
			flog.write('\n')

		#else, evaluate predictions
		else:
			plot_name=songs_data['Concert name'][i_song]+'_'+str(offset)
			scores_offset=[utils.eval_output(out_labels, out_probs, val_set['labels'].squeeze(),eval_tol, plot_savepath, plot_name)]
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
