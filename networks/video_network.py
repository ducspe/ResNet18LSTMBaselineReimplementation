import torch
import torch.nn as nn
import torchvision.models as models


# VIDEO only network
class VideoNet(nn.Module):

    def __init__(self, lstm_layers, lstm_hidden_size):
        super(VideoNet, self).__init__()

        resnet = models.resnet18(pretrained=False)  # set num_ftrs = 512

        num_ftrs = 512

        self.lstm_input_size = num_ftrs
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.features = nn.Sequential(
            *list(resnet.children())[:-1]  # drop the last FC layer
        )

        self.lstm_video = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=False)

        self.vad_video = nn.Linear(self.lstm_hidden_size, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        batch, frames, channels, height, width = x.size()
        # Reshape to (batch * seq_len, channels, height, width)
        x = x.view(batch*frames, channels, height, width)
        x = self.features(x).squeeze()  # output shape - Batch X Features X seq len
        x = self.dropout(x)
        # Reshape to (batch , seq_len, Features)
        x = x.view(batch, frames, -1)
        # Reshape to (seq_len, batch, Features)
        x = x.permute(1, 0, 2)

        h0 = torch.zeros(self.lstm_layers, batch, self.lstm_hidden_size).cuda()
        c0 = torch.zeros(self.lstm_layers, batch, self.lstm_hidden_size).cuda()

        out, _ = self.lstm_video(x, (h0, c0))  # output shape - seq len X Batch X lstm size
        out = self.dropout(out[-1])  # select last time step. many -> one
        out = torch.sigmoid(self.vad_video(out))
        return out

