## Unsupervised method (MATLAB)

### Usage
* To predict boundaries in a test audio, first extract its features and then run
```
predict_boundaries(path/to/features/filename.mat)
```
This function takes as input the path to the saved  ```.mat``` file containing the features, and displays the boundaries predicted by each of the features alone and by fusing the information from all of them (see Section 4.1 in the paper). </br></br>


* To visualise the extracted frame-wise features, rhythmogram, SDM and corresponding novelty curves (Figures 6, 7, 10 & 11 in the paper) for any concert in the dataset, run
```
case_study_analysis(song_ind,song_path)
```
where ```song_ind``` is the serial number of the song in the dataset, and ```song_path``` is the path to the folder containing the audio. </br></br>


* To reproduce the evaluation results in the paper, run the script
```
boundary_detection_evaluation(path/to/features)
```
