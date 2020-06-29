## CNN classifier
Using the implementation provided by [Keras](https://keras.io/) </br>

### Contents
```params.py```: Master file of all the hyperparameter values, directory and file paths, etc., and is imported by all other files. </br> </br>
```make_cv_dataset.py```: Generate the mel-spectrogram of each audio, divide into chunks of a specified context duration, and save each chunk as a ```.npy``` file. 
Each chunk is assigned a single label corresponding to the center frame. Labels are binary - 1 indicating boundary frame and 0 otherwise. 
Boundary labels are smeared \- all frames in a neighbourhood of the manually annotated boundary are also marked boundary. 
The labels are stored in a separate dictionary with the filenames of the saved chunks as keys, and are also saved as ```.npy``` files.
</br> </br>
```train_val_test```: To perform cross-validation on the 20-concert dataset and testing on the separate test set. Run with 1 or 0 as a command-line parameter - 
1 for cross-validation and 0 for testing. </br> </br>
