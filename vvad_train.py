from video_train_dataset import VideoDataset
from video_validation_dataset import TestValDataset
import torch
from networks.video_network import VideoNet
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import create_video_paths_list, create_audio_paths_list, MyLogger
import os

# Define parameters:
continue_training_initialization_checkpoint = ""
gpu_list = [0]  # [0, 1] # IDs of gpus to use for training
num_epochs = 1  # 10000
batch_size = 2  # 16
val_batch_size = 2  # 16
seq_length = 15  # 15
checkpoint_save_freq = 1  # 2
num_workers = 0  # For loading the data

lstm_layers = 2
lstm_hidden_size = 1024
learning_rate = 0.0001
epsilon = 1e-8

# Define model and dataset:
base_dir = "data_dda"
video_train_path = "{}/video/train".format(base_dir)
video_validation_path = "{}/video/dev".format(base_dir)
clean_audio_train_path = "{}/clean_audio/train".format(base_dir)
clean_audio_validation_path = "{}/clean_audio/dev".format(base_dir)

###########################################################################################
# End of configs section
###########################################################################################

my_logger = MyLogger("Train_")
video_train_paths_list = create_video_paths_list(video_train_path)
print("Video train paths: ", video_train_paths_list)

video_validation_paths_list = create_video_paths_list(video_validation_path)
print("Video validation paths: ", video_validation_paths_list)

clean_audio_train_paths_list = create_audio_paths_list(clean_audio_train_path)
print("Audio train paths: ", clean_audio_train_paths_list)

clean_audio_validation_paths_list = create_audio_paths_list(clean_audio_validation_path)
print("Audio validation paths: ", clean_audio_validation_paths_list)

assert len(video_train_paths_list) == len(clean_audio_train_paths_list)
assert len(video_validation_paths_list) == len(clean_audio_validation_paths_list)

model = VideoNet(lstm_layers=lstm_layers, lstm_hidden_size=lstm_hidden_size).cuda()
if continue_training_initialization_checkpoint:
    model.load_state_dict(torch.load(continue_training_initialization_checkpoint)["model_state_dict"])

model = torch.nn.DataParallel(model, device_ids=gpu_list)

os.makedirs('saved_models/checkpoints', exist_ok=True)
video_train_dataset = VideoDataset(video_paths=video_train_paths_list, label_paths=clean_audio_train_paths_list, seq_length=seq_length)
video_validation_dataset = TestValDataset(video_paths=video_validation_paths_list, label_paths=clean_audio_validation_paths_list, seq_length=seq_length)

train_loader = DataLoader(
    video_train_dataset,
    batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=False,
    drop_last=False
)

validation_loader = DataLoader(
    video_validation_dataset,
    batch_size=val_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False,
    drop_last=False
)


weight = torch.FloatTensor(2)
weight[0] = 1
weight[1] = 1
criterion = nn.CrossEntropyLoss(weight=weight).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))


def validation_routine():
    model.eval()
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []
    val_accuracy_list = []
    val_tnr_list = []
    val_loss_list = []
    with torch.no_grad():
        for val_batch_count, val_batch_data in enumerate(validation_loader):
            val_video_sequence, val_target_label_vad = val_batch_data
            val_video_sequence = val_video_sequence.cuda()
            val_target_label_vad = val_target_label_vad.cuda()
            val_prob_vad = model(val_video_sequence)

            val_loss = criterion(val_prob_vad, val_target_label_vad.long())
            val_loss_list.append(val_loss)

            _, val_predicted_vad = torch.max(val_prob_vad.data, 1)

            val_tp = (val_target_label_vad * val_predicted_vad).sum().type(torch.FloatTensor)
            val_tn = ((1 - val_target_label_vad) * (1 - val_predicted_vad)).sum().type(torch.FloatTensor)
            val_fp = ((1 - val_target_label_vad) * val_predicted_vad).sum().type(torch.FloatTensor)
            val_fn = (val_target_label_vad * (1 - val_predicted_vad)).sum().type(torch.FloatTensor)

            val_precision = val_tp / (val_tp + val_fp + epsilon)
            val_recall = val_tp / (val_tp + val_fn + epsilon)
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + epsilon)
            val_accuracy = (val_tp + val_tn) / (val_tp + val_tn + val_fp + val_fn + epsilon)
            val_tnr = val_tn / (val_tn + val_fp + epsilon)

            val_precision_list.append(val_precision)
            val_recall_list.append(val_recall)
            val_f1_list.append(val_f1)
            val_accuracy_list.append(val_accuracy)
            val_tnr_list.append(val_tnr)

    return sum(val_precision_list)/len(val_precision_list), sum(val_recall_list)/len(val_recall_list),\
           sum(val_f1_list)/len(val_f1_list), sum(val_accuracy_list)/len(val_accuracy_list),\
           sum(val_tnr_list)/len(val_tnr_list), sum(val_loss_list)/len(val_loss_list)


val_f1_forbestval = 0
val_loss_forbestval = 1e6
epoch_forbestval = 0
val_acc_forbestval = 0
val_prec_forbestval = 0
val_rec_forbestval = 0
val_tnr_forbestval = 0
for epoch in range(num_epochs):
    model.train()
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    tnr_list = []
    train_loss_list = []
    for batch_count, batch_data in enumerate(train_loader):
        model.zero_grad()
        video_sequence, target_label_vad = batch_data
        video_sequence = video_sequence.cuda()
        target_label_vad = target_label_vad.cuda()
        prob_vad = model(video_sequence)

        loss = criterion(prob_vad, target_label_vad.long())
        train_loss_list.append(loss)
        loss.backward()
        optimizer.step()

        # Calculate the evaluation metrics:
        _, predicted_vad = torch.max(prob_vad.data, 1)

        tp = (target_label_vad * predicted_vad).sum().type(torch.FloatTensor)
        tn = ((1 - target_label_vad) * (1 - predicted_vad)).sum().type(torch.FloatTensor)
        fp = ((1 - target_label_vad) * predicted_vad).sum().type(torch.FloatTensor)
        fn = (target_label_vad * (1 - predicted_vad)).sum().type(torch.FloatTensor)

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        tnr = tn / (tn + fp + epsilon)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
        tnr_list.append(tnr)

        info_line1 = "Epoch {}, batch {}: F1={:.5f}, Acc={:.5f}, Prec={:.5f}, Rec={:.5f}, TNR={:.5f}, Loss={:.5f}".format(epoch, batch_count, f1, accuracy, precision, recall, tnr, loss)
        print(info_line1)

    tr_len = len(train_loader)
    total_f1 = sum(f1_list)
    total_accuracy = sum(accuracy_list)
    total_precision = sum(precision_list)
    total_recall = sum(recall_list)
    total_tnr = sum(tnr_list)
    total_loss = sum(train_loss_list)

    info_line2 = "--------------START EVAL--------------------"
    print(info_line2)
    my_logger.log(info_line2)

    info_line3 = "Epoch {}: avg train F1={:.5f}, avg train Acc={:.5f}, avg train precision={:.5f}, avg train recall={:.5f}, avg train tnr={:.5f}, avg train loss={:.5f}".format(
        epoch, total_f1 / tr_len, total_accuracy / tr_len, total_precision / tr_len, total_recall / tr_len,
        total_tnr / tr_len, total_loss / tr_len)
    print(info_line3)
    my_logger.log(info_line3, extra_newline=True)
    # Evaluate on the validation dataset:
    vr_prec, vr_rec, vr_f1, vr_acc, vr_tnr, vr_loss = validation_routine()  # vr: validation routine
    info_line4 = "Avg validation F1={:.5f}, avg validation Acc={:.5f}, avg validation Prec={:.5f}, avg validation Rec={:.5f}, TNR={:.5f}, avg validation loss={:.5f}".format(
        vr_f1, vr_acc, vr_prec, vr_rec, vr_tnr, vr_loss)
    print(info_line4)
    my_logger.log(info_line4, extra_newline=True)
    if vr_loss < val_loss_forbestval:
        val_f1_forbestval = vr_f1
        val_loss_forbestval = vr_loss
        val_acc_forbestval = vr_acc
        val_prec_forbestval = vr_prec
        val_rec_forbestval = vr_rec
        val_tnr_forbestval = vr_tnr
        epoch_forbestval = epoch
        torch.save(model, "saved_models/best_model.pt")

    info_line5 = "The best validation session had f1 = {:.5f} and got registered at epoch {}. Accuracy at that point was {:.5f}, Prec was {:.5f}, Rec was {:.5f} TNR was {:.5f}, and loss was {:.5f}".format(
        val_f1_forbestval, epoch_forbestval, val_acc_forbestval, val_prec_forbestval, val_rec_forbestval,
        val_tnr_forbestval, val_loss_forbestval)
    print(info_line5)
    my_logger.log(info_line5)

    info_line6 = "--------------END EVAL----------------------"
    print(info_line6)
    my_logger.log(info_line6)
    # Save checkpoints:
    if epoch % checkpoint_save_freq == 0:
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        checkpoint_name = "epoch{}_valf1_{:.4f}_valacc_{:.4f}_valloss_{:.4f}_trainloss_{:.4f}_checkpoint.pt".format(
            epoch, vr_f1, vr_acc, vr_loss, total_loss / tr_len)
        torch.save(state, "saved_models/checkpoints/{}".format(checkpoint_name))
