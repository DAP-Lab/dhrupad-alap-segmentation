# Dhrupad Vocal Alap Segmentation
Annotation data of the structral segments of *alap* for a Dhrupad vocal concert dataset, and codes for automatic segmentation. This
repository is linked to the following publication: </br> </br>
```
First Author11, Second Author22, and Third Author31(2020). Structural Segmentation of Alap 
in Dhrupad Vocal Concerts, Transactions  of the International Society for 
Music Information Retrieval, Under Review
```

The annotations were created manually by one of the authors in consultation with a musician. Trained models are made available to obtain predictions on any test audio. Training scripts are also provided to reproduce the results reported in the paper.

### Contents
The [annotations](./annotations) folder contains annotations for the 20 concert cross-validation and 2 concert test datasets used in the paper. </br>
The [codes](./codes) folder contains the implementations of the feature extraction step and all the boundary detection methods as described in the paper. </br>
More details on the annotation format and running the codes can be found in the respective folders.

### Audio dataset
The sources for all the audios used in the work are listed in the file [Dataset_sources.pdf](./Dataset_sources.pdf). Some are available on YouTube, while others are from the CompMusic Dunya [1] collection and can be obtained through the Dunya API [2] using the provided MusicBrainz IDs [3]. </br>

### References
[1]: [https://dunya.compmusic.upf.edu/](https://dunya.compmusic.upf.edu/) </br>
[2]: [https://dunya.compmusic.upf.edu/developers/](https://dunya.compmusic.upf.edu/developers/) </br>
[3]: [https://musicbrainz.org/doc/MusicBrainz_Identifier](https://musicbrainz.org/doc/MusicBrainz_Identifier) </br>
