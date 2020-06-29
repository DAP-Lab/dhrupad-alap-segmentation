import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
import utils
from params import *
	
train_flag=int(sys.argv[1])		#1 or 0 indicating whether to run cross-valiation (1) or test (0)

##The cross-validation function (can be run in parallel on several folds using Multiprocessing)
def cross_val(fold):
	#Train
	song_inds_val=song_inds[fold*fold_size:(fold+1)*fold_size]
	train_set = utils.get_train_set(dataset, song_lengths, song_inds_val)
	clf = RandomForestClassifier(random_state=0, n_estimators=n_trees)
	clf.fit(train_set['features'],train_set['labels'][:,0])

	#save the model to disk
	save_model_path = os.path.join(model_savepath, 'model_fold%d.sav'%(fold))
	pickle.dump(clf, open(save_model_path, 'wb'))

	#Validate
	scores=[]
	for song_num in song_inds_val:
		val_set = utils.get_val_set(context_len, target_smear_width, feat_subset, features_matlab_path, song_num, songdata)
		out_probs=clf.predict_proba(val_set['features'])[:,1]
		
		for predict_thresh in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
			out_labels=(out_probs > predict_thresh).astype(int)
			plot_name='fold%d_'%fold+songdata['Concert name'][song_num]+'_thresh_'+str(predict_thresh)
			scores=np.append(scores, utils.eval_output(out_labels, out_probs, val_set['labels'][:,0], eval_tol, plot_savepath, plot_name))

	return scores

##Main function
if __name__=='__main__':	
	#read annotation data
	songdata=pd.read_csv(songdata_filepath)
	songdata_test=pd.read_csv(songdata_test_filepath)
	
	song_inds=np.arange(songdata.shape[0])
	n_folds=20
	fold_size=int(songdata.shape[0]/n_folds)

	#load entire cross-validation dataset
	dataset, song_lengths = utils.load_dataset(songdata, feat_subset, features_RF_path)

	#open file to log scores
	flog=open(os.path.join(log_savepath,'RF_train_test.txt'),'a')
	flog.write('----------feat_subset=%s\ttexwin_len=%d\tcontext_len=%d\tnTrees=%d\t-----------\n'%(feat_subset, texwin_len, context_len, n_trees))

	scores=[]
	
	if train_flag:
	##cross-validation

		flog.write('--Cross-validation scores--\n')
		for i_fold in range(n_folds):
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

	else:
		##Test
		flog.write('--Test scores--\n')

		#load the model from disk
		if not os.path.exists(test_model_path):
			print('Required model not available, performing training first.\n')
			cross_val(test_fold)
		clf = pickle.load(open(test_model_path, 'rb'))

		#loop over test songs
		for song_num in range(songdata_test.shape[0]):
			flog.write(songdata_test['Concert name'][song_num]+'\n')
			
			#if boundaries provided, read them
			try: boundaries=songdata_test['Boundaries'][song_num].split(',')
			except: boundaries=[]
			
			#loop over offset versions of test audio (test-time augmented)
			data_filepath=os.path.join(testfeatures_matlab_path,songdata_test['Concert name'][song_num])
			for offset in [0.1,0.2,0.3,0.4,0.5]:
				val_set=utils.get_val_set(context_len, target_smear_width, feat_subset, features_matlab_path, song_num=None, data_filepath=data_filepath, boundaries=boundaries, offset_list=[offset])
				out_probs=clf.predict_proba(val_set['features'])[:,1]
				out_labels=(out_probs > 0.5).astype(int)

				#if no boundaries available, plot output
				if len(boundaries)==0:
					plt.plot(out_probs); plt.plot(out_labels); plt.show()
				
				#else, evaluate predictions
				else:
					plot_name=songdata_test['Concert name'][song_num]+'_'+str(offset)
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
