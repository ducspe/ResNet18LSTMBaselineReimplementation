import librosa
from processing.target import clean_speech_VAD
import os
from datetime import datetime


# Define parameters:
# global_frame_rate = 29.970030  # frames per second
wlen_sec = 0.064  # window length in seconds
hop_percent = 0.25  # math.floor((1 / (wlen_sec * global_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
win = 'hann'  # type of window function (to perform filtering in the time domain)
center = False  # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect'  # This argument is ignored if center = False
pad_at_end = True  # pad audio file at end to match same size after stft + istft

# Noise robust VAD
vad_quantile_fraction_begin = 0.5  # 0.93
vad_quantile_fraction_end = 0.55  # 0.99
vad_quantile_weight = 1.0  # 0.999
vad_threshold = 1.7

# Noise robust IBM
ibm_quantile_fraction = 0.25  # 0.999
ibm_quantile_weight = 1.0  # 0.999
ibm_threshold = 50


# Other parameters:
sampling_rate = 16000
dtype = 'complex64'
eps = 1e-8


def create_ground_truth_labels_from_path(audio_path):
    raw_clean_audio, Fs = librosa.load(audio_path, sr=sampling_rate)

    mask_labels = clean_speech_VAD(raw_clean_audio,
                           fs=sampling_rate,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    return mask_labels.T


def create_ground_truth_labels(raw_clean_audio):

    mask_labels = clean_speech_VAD(raw_clean_audio,
                           fs=sampling_rate,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    return mask_labels.T


def create_video_paths_list(base_path):
    video_paths_list = []
    speaker_folders = sorted([x for x in os.listdir(base_path)])
    for speaker in speaker_folders:
            speaker_path = os.path.join(base_path, speaker)
            speaker_mat_files = sorted([y for y in os.listdir(speaker_path)])

            for sentence_mat_file in speaker_mat_files:
                sentence_video_path = os.path.join(speaker_path, sentence_mat_file)
                video_paths_list.append(sentence_video_path)

    return video_paths_list


def create_audio_paths_list(base_path):
    audio_paths_list = []
    speaker_folders = sorted([x for x in os.listdir(base_path)])
    for speaker in speaker_folders:
            speaker_path = os.path.join(base_path, speaker, "straightcam")
            speaker_wav_files = sorted([y for y in os.listdir(speaker_path)])

            for sentence_wav_file in speaker_wav_files:
                sentence_audio_path = os.path.join(speaker_path, sentence_wav_file)
                audio_paths_list.append(sentence_audio_path)

    return audio_paths_list


class MyLogger:
    def __init__(self, prefix):
        os.makedirs("logs_folder", exist_ok=True)
        logs = open(f"logs_folder/{prefix}loggerfile_{datetime.now()}.txt", "w")
        self.name = logs.name
        logs.write(f"The logging started at: {datetime.now()}\n\n")
        logs.close()

    def log(self, my_message, extra_newline=False):
        logs = open(self.name, "a")
        logs.write(f"{my_message}\n")
        if extra_newline:
            logs.write("\n")
        logs.close()
