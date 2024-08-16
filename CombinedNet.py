import torch
from torch import nn
from torch.nn import functional as F
from UResIncNet import UResIncNet


def init_weights(module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        torch.nn.init.kaiming_normal_(module.weight)
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight)
        torch.nn.init.zeros_(module.bias)


class RNN(nn.Module):

    def __init__(self, nclass=5, in_chans=1, lstm_max_features=512, lstm_layers=2,
                 lstm_dropout: float = 0.1,
                 lstm_bidirectional=True, lstm_depth=1, custom_weight_init=False):
        """
        :param nclass: output channels
        :param in_chans: input channels
        :param lstm_max_features: hidden lstm units
        :param lstm_layers: number of stacked lstm layers
        :param lstm_dropout:
        :param lstm_bidirectional: boolean
        :param lstm_depth: Depreciated
        :param custom_weight_init:
        """
        super().__init__()
        self.nclass = nclass
        self.in_chans = in_chans
        self.lstm_max_features = lstm_max_features
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_depth = lstm_depth

        self.lstms = nn.ModuleList()
        hidden_size = lstm_max_features // 2 if lstm_bidirectional else lstm_max_features
        for _ in range(lstm_depth):
            self.lstms.append(nn.LSTM(input_size=in_chans, hidden_size=hidden_size, num_layers=lstm_layers,
                                      bidirectional=lstm_bidirectional, dropout=lstm_dropout, batch_first=True))
            # outputs: (BatchSize,SeqLength, max_features)
        out_chans = lstm_max_features

        self.logits = nn.Conv1d(out_chans, nclass, kernel_size=1, stride=1)

        if custom_weight_init:
            self.apply(init_weights)

    def forward(self, x):
        # x is (N, C, L)
        # LSTM expects: (N, L, C)
        x = torch.swapaxes(x, 1, 2)

        for lstm in self.lstms:
            x, _ = lstm(x)

        # x is (N, L, C)
        # Conv expects: (N, C, L)
        x = torch.swapaxes(x, 1, 2)

        # Return the logits:
        return self.logits(x)

    def get_args_summary(self):
        # For backwards compatibility with older class which did not have layers attribute:

        return (f"lstm_MaxCH {self.max_channels} - lstm_Depth {self.lstm_depth} - "
                f"lstm_Bidirectional {self.lstm_bidirectional} "
                f"- lstm_Layers {self.lstm_layers} - lstm_Dropout {self.lstm_dropout}")

    def get_kwargs(self):
        kwargs = {"nclass": self.nclass, "in_chans": self.in_chans,
                  "lstm_max_channels": self.max_channels, "lstm_layers": self.lstm_layers,
                  "lstm_dropout": self.lstm_dropout, "lstm_bidirectional": self.lstm_bidirectional}
        return kwargs


class CombinedNet(nn.Module):
    def __init__(self, nclass=5, in_size=512, in_chans=1, max_channels=512, depth=8, kernel_size=4, layers=1,
                 sampling_factor=2,
                 sampling_method="conv_stride", dropout=0.0, skip_connection=True,
                 lstm_max_features=512, lstm_layers=2, lstm_dropout: float = 0.1, lstm_bidirectional=True,
                 lstm_depth=1, custom_weight_init=False, split_channels_into_branches=False):
        super().__init__()
        self.nclass = nclass
        self.in_size = in_size
        self.in_chans = in_chans
        self.max_channels = max_channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.layers = layers
        self.sampling_factor = sampling_factor
        self.sampling_method = sampling_method
        self.dropout = dropout
        self.skip_connection = skip_connection

        self.lstm_max_features = lstm_max_features
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_depth = lstm_depth
        self.split_channels_into_branches = split_channels_into_branches

        if split_channels_into_branches:
            assert in_chans==2
            # One channel for UResIncNet and one for RNN
            in_chans = 1
        

        self.UResIncNetBranch = UResIncNet(nclass=nclass, in_chans=in_chans, max_channels=max_channels, depth=depth,
                                           kernel_size=kernel_size, layers=layers,
                                           sampling_factor=sampling_factor, sampling_method=sampling_method,
                                           dropout=dropout, skip_connection=skip_connection)
        self.LSTMBranch = RNN(nclass=nclass, in_chans=in_chans, lstm_max_features=lstm_max_features,
                              lstm_layers=lstm_layers,
                              lstm_dropout=lstm_dropout,
                              lstm_bidirectional=lstm_bidirectional, lstm_depth=lstm_depth)

        self.logits = nn.Sequential(
            nn.BatchNorm1d(2 * nclass),
            nn.LeakyReLU(0.2),
            nn.Conv1d(2 * nclass, nclass, kernel_size=1, stride=1)
        )

        if custom_weight_init:
            self.apply(init_weights)

    def forward(self, x):
        if self.split_channels_into_branches:
            x_pleth = x[:,[0], :]
            x_spo2 = x[:,[1], :]
            x1 = self.UResIncNetBranch(x_pleth)
            x2 = self.LSTMBranch(x_spo2)
        else:
            x1 = self.UResIncNetBranch(x)
            x2 = self.LSTMBranch(x) 
        x = torch.cat((x1, x2), dim=1)

        # Return the logits
        return self.logits(x)

    def get_args_summary(self):
        # For backwards compatibility with older class which did not have layers attribute:
        if hasattr(self, "layers"):
            layers = self.layers
        else:
            layers = 1

        if hasattr(self, "dropout"):
            dropout = self.dropout
        else:
            dropout = 0.0

        if hasattr(self, "split_channels_into_branches"):
            split_channels_into_branches = self.split_channels_into_branches
        else:
            split_channels_into_branches = False
        
        return (f"MaxCH {self.max_channels} - Depth {self.depth} - Kernel {self.kernel_size} "
                f"- Layers {layers} - Sampling {self.sampling_method} - Dropout {dropout}"
                f"- lstm_MaxCH {self.max_channels} - "
                f"lstm_Bidirectional {self.lstm_bidirectional} "
                f"- lstm_Layers {self.lstm_layers} - lstm_Dropout {self.lstm_dropout} - split_channels_into_branches {split_channels_into_branches}")

    def get_kwargs(self):
        # For backwards compatibility with older class which did not have layers attribute:
        if hasattr(self, "layers"):
            layers = self.layers
        else:
            layers = 1

        if hasattr(self, "dropout"):
            dropout = self.dropout
        else:
            dropout = 0.0

        if hasattr(self, "split_channels_into_branches"):
            split_channels_into_branches = self.split_channels_into_branches
        else:
            split_channels_into_branches = False
        
        kwargs = {"nclass": self.nclass, "in_size": self.in_size, "in_chans": self.in_chans,
                  "max_channels": self.max_channels,
                  "depth": self.depth, "kernel_size": self.kernel_size, "layers": layers,
                  "sampling_factor": self.sampling_factor, "sampling_method": self.sampling_method,
                  "dropout": dropout, "skip_connection": self.skip_connection,
                  "lstm_max_channels": self.max_channels, "lstm_layers": self.lstm_layers,
                  "lstm_dropout": self.lstm_dropout, "lstm_bidirectional": self.lstm_bidirectional, "split_channels_into_branches": split_channels_into_branches
                  }
        return kwargs
