from video_validation_dataset import TestValDataset
from networks.video_network import VideoNet
from utils import create_video_paths_list, create_audio_paths_list
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import MyLogger

# Parameters:
test_gpu_list = [0]
model_to_evaluate = "saved_models/best_model.pt"
# model_to_evaluate = "saved_models/checkpoints/checkpointfile"

base_test_dir = "data_dda"
test_video_path = "{}/video/test".format(base_test_dir)
test_clean_audio_path = "{}/clean_audio/test".format(base_test_dir)

test_num_workers = 0
test_batch_size = 3
test_seq_length = 2

lstm_layers = 2
lstm_hidden_size = 1024
epsilon = 1e-8

############################################################
# End of configuration section
############################################################
my_test_logger = MyLogger("Test_")

test_video_paths_list = create_video_paths_list(test_video_path)
test_clean_audio_paths_list = create_audio_paths_list(test_clean_audio_path)

test_model = VideoNet(lstm_layers=lstm_layers, lstm_hidden_size=lstm_hidden_size).cuda()
test_model = torch.nn.DataParallel(test_model, device_ids=test_gpu_list)

if model_to_evaluate.__contains__("_checkpoint"):
    print("Loading the checkpoint")
    test_model.load_state_dict(torch.load(model_to_evaluate)["model_state_dict"])
else:
    print("Loading the best model")
    test_model = torch.load(model_to_evaluate)

test_model.eval()  # turn on inference mode

video_test_dataset = TestValDataset(video_paths=test_video_paths_list, label_paths=test_clean_audio_paths_list, seq_length=test_seq_length)

test_loader = DataLoader(
    video_test_dataset,
    batch_size=test_batch_size, shuffle=False,
    num_workers=test_num_workers, pin_memory=False,
    drop_last=False
)

weight = torch.FloatTensor(2)
weight[0] = 1
weight[1] = 1
test_criterion = nn.CrossEntropyLoss(weight=weight).cuda()

test_precision_list = []
test_recall_list = []
test_f1_list = []
test_accuracy_list = []
test_tnr_list = []
test_loss_list = []

with torch.no_grad():
    for test_batch_count, test_batch_data in enumerate(test_loader):
        test_video_sequence, test_target_label_vad = test_batch_data
        test_video_sequence = test_video_sequence.cuda()
        test_target_label_vad = test_target_label_vad.cuda()
        test_prob_vad = test_model(test_video_sequence)

        test_loss = test_criterion(test_prob_vad, test_target_label_vad.long())
        test_loss_list.append(test_loss)

        _, test_predicted_vad = torch.max(test_prob_vad.data, 1)
        test_tp = (test_target_label_vad * test_predicted_vad).sum().type(torch.FloatTensor)
        test_tn = ((1 - test_target_label_vad) * (1 - test_predicted_vad)).sum().type(torch.FloatTensor)
        test_fp = ((1 - test_target_label_vad) * test_predicted_vad).sum().type(torch.FloatTensor)
        test_fn = (test_target_label_vad * (1 - test_predicted_vad)).sum().type(torch.FloatTensor)

        test_precision = test_tp / (test_tp + test_fp + epsilon)
        test_recall = test_tp / (test_tp + test_fn + epsilon)
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + epsilon)
        test_accuracy = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn + epsilon)
        test_tnr = test_tn / (test_tn + test_fp + epsilon)

        test_precision_list.append(test_precision)
        test_recall_list.append(test_recall)
        test_f1_list.append(test_f1)
        test_accuracy_list.append(test_accuracy)
        test_tnr_list.append(test_tnr)

test_avg_f1 = sum(test_f1_list)/len(test_f1_list)
test_avg_acc = sum(test_accuracy_list)/len(test_accuracy_list)
test_avg_prec = sum(test_precision_list)/len(test_precision_list)
test_avg_rec = sum(test_recall_list)/len(test_recall_list)
test_avg_tnr = sum(test_tnr_list)/len(test_tnr_list)
test_avg_loss = sum(test_loss_list)/len(test_loss_list)

log_info = "Avg test F1={:.5f}, avg test Acc={:.5f}, avg test Prec={:.5f}, avg test Rec={:.5f}, TNR={:.5f}, avg test loss={:.5f}".format(test_avg_f1, test_avg_acc, test_avg_prec, test_avg_rec, test_avg_tnr, test_avg_loss)

print(log_info)
my_test_logger.log(log_info)
