This is a reimplementation of the baseline code for our paper <em>"See the silence: improving visual-only voice activity detection by optical flow and RGB fusion"</em> available at: https://github.com/ducspe/VVADpaper <br><br>
This baseline is compared to our new modified approach available at: https://github.com/ducspe/VisualOnlyVoiceActivityDetection. <br><br>
The ResNet and LSTM architecture from the original authors was faithfully kept. The supporting infrastructure code around it was changed however to compress the original code significantly, since we do not need the audio branch and the audio-visual version present in the original implementation. <br><br>
Unlike in the original implementation, we do not label manually, but rather implement a separate module that infers ground truth labels from the audio stream. Audio related code is available in the **processing** folder<br><br>
The **data_dda** folder contains a subset of the preprocessed TCD-TIMIT dataset. The structure of this folder must be kept when working with the full dataset. <br><br>
The preprocessed inputs can be obtained by using the code from: https://github.com/ducspe/TCD-TIMIT-Preprocessing 
