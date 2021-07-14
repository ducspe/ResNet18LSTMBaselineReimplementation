import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms
import utils
import librosa
# import skvideo.io
# import os

# params:
epsilon = 1e-8  # for numerical stability
sampling_rate = 16000


class TestValDataset(Dataset):
    def __init__(self, video_paths, label_paths, seq_length):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.seq_length = seq_length

        self.video_train_info = np.load('video_train_statistics.npy', allow_pickle=True)
        self.video_train_normalization_mean = self.video_train_info.item()["all_videos_mean_before_normalization"]
        self.video_train_normalization_std = self.video_train_info.item()["all_videos_std_before_normalization"]

        self.video_ram = []
        self.clean_label_ram = []

        for video_pa, clean_audio_pa in zip(video_paths, self.label_paths):
            preprocessed_file = np.load(video_pa) / 255.0
            clean_audio, Fs = librosa.load(clean_audio_pa, sr=sampling_rate)
            clean_audio_label = utils.create_ground_truth_labels(clean_audio)
            sync_len = min(preprocessed_file.shape[0], clean_audio_label.shape[0])

            self.video_ram.append(preprocessed_file[:sync_len])
            self.clean_label_ram.append(clean_audio_label[:sync_len])

        self.video_ram_concat = np.concatenate(self.video_ram, axis=0)
        self.clean_label_ram_concat = np.concatenate(self.clean_label_ram, axis=0)

        print("The shape of video_ram_concat is: ", self.video_ram_concat.shape)
        print("The shape of clean_label_ram_concat is: ", self.clean_label_ram_concat.shape)

    def __len__(self):
        return len(self.video_ram_concat) - self.seq_length

    def __getitem__(self, id):
        # Video
        normalize = torchvision.transforms.Normalize(
            # mean=[self.video_train_normalization_mean, self.video_train_normalization_mean,
            #       self.video_train_normalization_mean],
            # std=[self.video_train_normalization_std, self.video_train_normalization_std,
            #      self.video_train_normalization_std]
            mean=[0, 0, 0],
            std=[1, 1, 1]
        )

        transform_forward = torchvision.transforms.Compose([
            normalize,
        ])

        video_snip = self.video_ram_concat[id:id+self.seq_length]

        video_sample = torch.FloatTensor(self.seq_length, 3, 67, 67)
        for snip_count, frame_snipped in enumerate(video_snip):
            video_sample[snip_count] = torch.from_numpy(np.stack((video_snip[snip_count],) * 3, axis=0))
            # from single channel gray to 3-channel gray

        augmented_video_sample = transform_forward(video_sample)

        # Label:
        label = torch.squeeze(torch.from_numpy(self.clean_label_ram_concat[id+self.seq_length]), dim=0).type(torch.FloatTensor)

        # Uncomment below to test correctness of augmentation:
        # augmented_video_sample_numpy = augmented_video_sample.cpu().numpy()
        #
        # os.makedirs(f'mp4_dataout/', exist_ok=True)
        # dda_mp4_file = f'mp4_dataout/check{id}.mp4'
        # upsampling_video_writer = skvideo.io.FFmpegWriter(dda_mp4_file,
        #                                                   inputdict={'-r': str(62.5),
        #                                                              '-s': '{}x{}'.format(67, 67)},
        #                                                   outputdict={'-filter:v': 'fps=fps={}'.format(
        #                                                       str(62.5)),
        #                                                       '-c:v': 'libx264',
        #                                                       '-crf': str(17),
        #                                                       '-preset': 'veryslow'}
        #                                                   )
        #
        # for up_frame in augmented_video_sample_numpy:
        #     upsampling_video_writer.writeFrame(up_frame*255.0)
        #
        # upsampling_video_writer.close()
        # End of testing for correctness of augmentation code

        return augmented_video_sample.cuda(), label.cuda()
