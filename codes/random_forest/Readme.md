## Random Forest classifier
Using the implementation provided by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#) </br>

### Contents
```params.py```: master file of all the hyperparameter values, directory and file paths, etc., and is imported by all other files. </br> </br>
```make_cv_dataset.py```: takes the features extracted using the MATLAB codes, makes the different feature subsets (as in the paper), and creates a dictionary
for each song containing the frame-wise features and target labels. The labels are binary - 1 indicating boundary frame and 0 otherwise. Boundary labels are smeared 
\- all frames in a neighbourhood of the manually annotated boundary are also marked boundary. The dictionaries are stored as ```.npy``` files. </br> </br>
```train_val_test```: to perform cross-validation on the 20-concert dataset and testing on the separate test set. Run with 1 or 0 as a command-line parameter - 
1 for cross-validation and 0 for testing. </br> </br>
