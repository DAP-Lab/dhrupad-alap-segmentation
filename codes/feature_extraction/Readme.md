## Feature extraction codes (in MATLAB)

```extract_features.m``` - Extracts frame-level features from a specified audio and saves them in a structure as a ```.mat``` file. To extract features for an audio file, simply run </br>
```
extract_features('path/to/file/filename.wav')
```
</br>

Features get saved in a folder called ```features_from_matlab``` in a ```features``` folder (created by the script) above the codes directory. The saved structure contains the following fields: </br>
*  ```tempo_sal``` - Tempo and salience
*  ```rhythm_features``` - Posteriors of tempo and salience
*  ```ste_sc_diff``` - Biphasic filtered short-time energy and short-time spectral centroid
*  ```avg_MFCC``` - Mel-Frequency Cepstral Coefficients

All features are calculated at a short-time frame level and then averaged using longer overlapping windows of size 3s with a hop of 1s. These are the default values as reported in the paper, but can be modified using commandline arguments (see below). Every feature is stored as a column vector with each column corresponding to a 1s frame. The ```ste_sc_diff``` and ```avg_MFCC``` together constitute the **timbre** features. Refer to Figure 9 of the paper for a detailed block diagram of the feature extraction process.</br>

### Additional details
Full syntax - ```extract_features(in_path,Name,Value)``` </br>
```in_path``` is either the name of an audio file or a text file containing a column-list of audio filenames. It is a mandatory argument. </br>
Additional optional arguments can be specified using Name, Value pairs, and include the following:

* ```in_dir``` - path to the audio file(s) if ```in_path``` is only the name of the audio file without the full path. Useful if ```in_path``` is a text file with only the names of audios.
* ```tex_win``` - averaging window length in seconds (defauls to 3)
* ```tex_hop``` - hop in seconds between averaging windows (defaults to 1) 
* ```offsets``` - list of values in seconds to shift the audio in time by, as a way of performing test-time augmentation (offers a robust way of measuring test performance)
* ```out_path``` - path to save features (defaults to ```../../features/features_from_matlab/```, relative to the current directory)
