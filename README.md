## This is the baseline code to compare my new approach to. The ResNet and LSTM architecture from the original authors was faithfully kept. The supporting infrastructure code around it was changed however to compress the original code significantly, since we do not need the audio branch and the audio-visual version present in the original implementation.
## Unlike in the original implementation, we do not label manually, but rather implement a separate module that infers ground truth labels from the audio stream.
## The data_dda folder contains a subset of the preprocessed TCD-TIMIT dataset. The structure of this folder must be kept when working with the full dataset.
## The preprocessed inputs can be obtained by using the code from: https://github.com/rescer/TCD-TIMIT-Preprocessing 
