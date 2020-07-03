### Contents
[feature_extraction](feature_extraction) - MATLAB codes to extract feaures for use with the unsupervised and random forest methods. </br>
[unsupervised](unsupervised) - MATLAB codes for unsupervised SDM-novelty based boundary detection </br>
[random_forest](random_forest) - Python codes for the random forest implementation. </br>
[cnn](cnn) - Python codes for the CNN implementation. </br>

For the RF and CNN methods, models trained on the 20-concert dataset are also provided, and can be used to obtain predictions on any test audio.

### Description
To obtain boundary predictions for an audio file:
1. First extract all the features using the ```extract_features``` MATLAB function (more details inside [feature_extraction](./feature_extraction))
2. To use the unsupervised method, run ...
```

```
3. To use the trained RF model, run 
```
test_rf.py path/to/features/filename.mat
```
4. To use the trained CNN model, run 
```
test_cnn.py path/to/audio/filename.wav
```
