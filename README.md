Code for the publication:
Kritsiolis, D. and Kotropoulos, C. (2024). Mobile Phone Identification from Recorded Speech Signals Using Non-Speech Segments 
and Universal Background Model Adaptation. In Proceedings of the 13th International Conference on Pattern Recognition Applications 
and Methods - ICPRAM; ISBN 978-989-758-684-2; ISSN 2184-4313, SciTePress, pages 793-800. DOI: 10.5220/0012420400003654

Run the command addpath(genpath(pwd)) on the source directory before running the scripts.

For the following folders and scripts there may exist an identical folder/script with a 2 appended to its name. These 
folder/scripts refer to the ccnu_mobile dataset whereas the ones without 2 refer to the mobiphone dataset.

The following terms refer to:
 1. Audio frames: all the frames of the unaltered speech signal
 2. Non speech frames: all the frames of the speech signal after applying voice activity detection and removing
    all the speech samples
 3. Baseline: the initial MOBIPHONE dataset
 4. Gaussian: the baseline dataset plus the same dataset augmented with gaussian noise
 5. Background: the baseline dataset plus the same dataset augmented with background noise
 6. Reverberation: the baseline dataset plus the same dataset augmented with reverberation
 7. Augmented: the baseline dataset plus the same dataset with gaussian, background and reverberation augmentations
 8. Crop: the baseline dataset plus the same dataset with random croppings in the middle of the signals
 9. Loudness: the baseline dataset plus the same dataset with randomly changed loudness level
10. Pitch: the baseline dataset plus the same dataset with randomly changed pitch
11. Speed: the baseline dataset plus the same dataset with randomly changed speed
12. VTLP: the baseline dataset plus the same dataset with vocal tract length perturbation applied
13. Augmented2: the baseline dataset plus the same dataset with crop, loudness, pitch, speed and VTLP augmentations
14. All: the baseline dataset plus all the above individual augmentations

The "data" folders within the Code folder are not to be confused with the Data folder containing the speech databases.
From now on when the data folders are referenced, we mean the data folders within the Code folder.

The first script that should be executed is the make_dirs.m script. It creates the data folders and all the
needed subfolders in which the various features and models like the MFCCs and the GMMs will be stored and
accessed by the rest of the scripts (in case data folder does not exist).

The data folder has 3 subfolders for the MFCCs, the GMMs and the SVMs.
1. MFCCs\{1}\{2}\{3}\{4}
   {1} refers to the type of frames used (audio/non-speech)
   {2} refers to the dataset used
   {3} refers to the train\test set
   {4} refers to the phone the MFCCs belong to (one of the 21 phone IDs)
2. Brand and model dependent GMMs: data\GMMs\{1}\{2}\{3}
   {1} refers to the type of frames used (audio/non-speech)
   {2} refers to the dataset used
   {3} refers to the number of components of the GMM (1/2/4/8/16/32/64)
3. Speaker dependent GMMs: data\GMMs\{1}\speakers\{2}\{3}
   {1} refers to the type of frames used (audio/non-speech)
   {2} refers to the number of components of the GMM (1/2/4/8/16/32/64)
   {3} refers to the phone model (one of the 21 phone IDs)
4. SVMs: data\SVMs\{1}\{2}\{3}
   {1} refers to the type of frames used (audio/non-speech)
   {2} refers to the dataset used
   {3} refers to the number of components of the GMM from which the GSVs are extracted (1/2/4)

The augment_mobiphone.py script is a python script that augments the wav files of the MOBIPHONE dataset. It performs
3 kind of augmentations: addition of gaussian noise, addition of background noise (chattering in a restaurant) and
addition of reverberation. The gaussian and background noise are added to the wav files with an SNR randomly chosen
between 10db and 20db. It requires the MOBIPHONE dataset to exist in the same directory as the script.

The augment_mobiphone2.py script is a python script that augments the wav files of the MOBIPHONE dataset. It performs
5 kind of augmentations: random croppings, random loudness level adjustment, random pitch changes, random speed changes
and random vocal tract length perturbation. It requires the MOBIPHONE dataset to exist in the same directory as the 
script.

In the mfcc_extraction folder there are the extract_mfccs_{...}.m scripts. The extract_mfccs_{...}.m scripts are used 
to extract the MFCCs of the speech and non-speech frames of the recordings of both the training and the testing speakers
and then save those extracted features in the data folder. They extract the MFCCs of each speaker of each phone of each 
subset of the augmented dataset and save them to their corresponding folder in data.

In the gmm_training folder there are the train_gmms_{...}_{...}.m scripts. The train_gmms_{...}_{..}.m scripts use the 
corresponding extracted MFCCs (speech/non-speech) and construct model-dependent and brand-dependent GMMs as the training
models of the Maximum Likelihood classificaton and a GMM for each train and test speaker file, from which GMMs the GSVs 
will be extracted and used in the SVM classification. The GMMs are saved in the data folder. The word after the type of
frames in the name of the script refers to the subset used. If that word is speakers then the script trains the speaker 
dependent GMMs.

In the ml_classification folder there are the ML_classifier_{...}_{...}_{...}.m scripts. 
The ML_classifier_{...}_{...}_{...}.m scripts classify the test recordings into models/brands using the corresponding
extracted MFCCs (speech/non-speech) and the corresponding trained GMMs. The algorithm used is Maximum Likelihood 
classification using the negative log likelihood. The last word in the name of the script refers to the subset used.

In the svm_training folder there are the train_svms_{...}_{...}.m scripts. The train_svms_{...}_{...}.m scripts use the 
corresponding speaker GMMs trained on the speech/non-speech MFCCs to extract the GSVs (with means only and with means and
covariance) and train SVMs using bayesian optimization. The last word in the name of the script refers to the subset 
used.

In the svm_classification folder there are the SVM_classifier_{...}_{...}_{...}.m scripts.
The SVM_classifier_{...}_{...}_{...}.m scripts use the speech/non-speech GSV SVMs to classify each speaker recording to 
models/brands. The last word in the name of the script refers to the subset used.

In the neural_classification and zeng_paper_code folders there are the py scripts that use the proposed neural architectures
to perform mobile phone identification on the ccnu_mobile dataset. In the first folder the classification is done using
our data whereas in the second folder the classification is done using the data provided by the paper's authors (namely
the 64 component UBM they used). In the same folders there also are the make_train_test_sets scripts which use the 
corresponding data in the data2 folder to create the train and test .mat files that will be used by the py scripts. The
python scripts use the keras/tensorflow libraries and also require the NVIDIA CUDA libraries because the training of the
neural networks is done on the GPU.

In the functions folder there are various implemented audio and speech processing functions some of which were used. The
map_adapt_means.m script was written to implement mean vector MAP adaptation of the data. It receives as input a
gmdistribution object representing the GMM we MAP adapt, the data matrix of size num_examples x num_features and the
relevance factor of the adaptation coefficient. It outputs a num_components x num_feature matrix containing the adapted
mean vectors.

P.S. For the Matlab scripts relating to MFCC extraction the desired databases must be located in the source folder of the
code to run. The GMM and SVM training scripts require the data and data2 folders to be in the source directory and to 
contain the extracted MFCCs. The scripts relating to classification require the data folders to be located in the source
code directory and to contain the MFCCs, the trained GMMs and the trained SVMs.