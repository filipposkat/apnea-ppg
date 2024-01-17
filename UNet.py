import torch
from torch import nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    # Consists of Conv -> LeakyReLU(0.2) -> MaxPool
    def __init__(self, in_chans, out_chans, layers=2, kernel_size=3, sampling_factor=2, sampling_method="pooling"):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = layers
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        self.sampling_method = sampling_method

        self.encoder = nn.ModuleList()

        for _ in range(layers):
            self.encoder.append(nn.Conv1d(in_chans, out_chans, kernel_size=kernel_size, stride=1, padding="same"))
            self.encoder.append(nn.BatchNorm1d(out_chans))
            self.encoder.append(nn.LeakyReLU(0.2))
            in_chans = out_chans

        if sampling_method == "conv_stride" and sampling_factor > 1:
            diff = (kernel_size - sampling_factor)
            if diff % 2 == 0:
                ks = kernel_size
                pad = diff // 2
            else:
                ks = kernel_size + 1
                pad = (ks - sampling_factor) // 2

            self.encoder.append(nn.Sequential(
                nn.Conv1d(out_chans, out_chans, kernel_size=ks,
                          stride=sampling_factor, padding=pad),
                nn.BatchNorm1d(out_chans),
                nn.LeakyReLU(0.2)))
        elif sampling_method == "pooling" and sampling_factor > 1:
            self.encoder.append(nn.MaxPool1d(sampling_factor))

    def forward(self, x):
        for enc in self.encoder:
            x = enc(x)
        return x


class DecoderBlock(nn.Module):
    # Consists of 2x2 transposed convolution -> Conv -> LeakyReLU(0.2)
    def __init__(self, in_chans, out_chans, layers=2, kernel_size=3, skip_connection=True, sampling_factor=2,
                 dropout=0.0):
        super().__init__()
        self.skip_connection = skip_connection
        self.layers = layers
        self.kernel_size = kernel_size
        self.padding = "same"
        self.dropout = dropout

        skip_factor = 1 if skip_connection else 2
        self.decoder = nn.ModuleList()

        # Transpose convolution:
        # Lout = (Lin-1)*stride + dilation*(kernel-1) - 2*padding + output_padding + 1
        # if dilation = 1 then:
        # Lout = Lin*stride + kernel - stride - 2*padding + output_padding
        # So we control Lout by stride, and we want the "kernel - stride - 2*padding + output_padding" to be zero
        # In order to achieve this we want to set padding and output padding in a way that offsets (kernel-stride).
        # For example if stride == kernel then we set padding=output_padding =0

        if kernel_size == sampling_factor:
            pad = 0
            out_pad = 0
        elif (kernel_size - sampling_factor) % 2 == 0:
            pad = (kernel_size - sampling_factor) // 2
            out_pad = 0
        else:
            out_pad = 1
            pad = (kernel_size - sampling_factor + out_pad) // 2

        assert (kernel_size - sampling_factor - 2 * pad + out_pad) == 0
        self.tconv = nn.ConvTranspose1d(in_chans, in_chans // 2, stride=sampling_factor, kernel_size=kernel_size,
                                        padding=pad, output_padding=out_pad)

        self.decoder.append(nn.Conv1d(in_chans // skip_factor, out_chans, kernel_size, 1, padding="same"))
        self.decoder.append(nn.BatchNorm1d(out_chans))
        self.decoder.append(nn.LeakyReLU(0.2))
        for _ in range(layers - 1):
            self.decoder.append(nn.Conv1d(out_chans, out_chans, kernel_size, 1, padding="same"))
            self.decoder.append(nn.BatchNorm1d(out_chans))
            self.decoder.append(nn.LeakyReLU(0.2))

        if dropout > 0.0:
            self.decoder.append(nn.Dropout(p=dropout))

    def forward(self, x, enc_features=None):
        x = self.tconv(x)
        if self.skip_connection:
            x = torch.cat((enc_features, x), dim=1)
        for dec in self.decoder:
            x = dec(x)
        return x


class UNet(nn.Module):
    def __init__(self, nclass=1, in_chans=1, max_channels=512, depth=5, layers=2, kernel_size=3, sampling_factor=2,
                 sampling_method="pooling", skip_connection=True):
        """
        :param nclass:
        :param in_chans:
        :param max_channels:
        :param depth:
        :param layers:
        :param kernel_size:
        :param sampling_factor:
        :param sampling_method: either "pooling" or "conv_stride"
        :param skip_connection:
        """
        super().__init__()
        self.nclass = nclass
        self.in_chans = in_chans
        self.max_channels = max_channels
        self.depth = depth
        self.layers = layers
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        self.sampling_method = sampling_method
        self.skip_connection = skip_connection

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        first_out_chans = max_channels // (2 ** (depth - 1))
        if first_out_chans % 4 == 0:
            out_chans = first_out_chans
        else:
            out_chans = 4

        # The first block should not do any down-sampling (stride = 1 and no pooling):
        self.encoder.append(EncoderBlock(in_chans, out_chans, layers, kernel_size=kernel_size,
                                         sampling_factor=1, sampling_method="no sampling"))

        for _ in range(depth - 1):
            if out_chans * 2 <= max_channels:
                in_chans, out_chans = out_chans, out_chans * 2
            else:
                in_chans, out_chans = out_chans, out_chans
            self.encoder.append(EncoderBlock(in_chans, out_chans, layers=layers, kernel_size=kernel_size,
                                             sampling_factor=sampling_factor, sampling_method=sampling_method))

        for _ in range(depth - 1):
            if out_chans // 2 >= 4:
                in_chans, out_chans = out_chans, out_chans // 2
            else:
                in_chans, out_chans = out_chans, out_chans
            self.decoder.append(DecoderBlock(in_chans, out_chans, layers=layers, kernel_size=kernel_size,
                                             sampling_factor=sampling_factor))

        # Add a 1x1 convolution to produce final classes
        self.logits = nn.Conv1d(out_chans, nclass, 1, 1)

    def forward(self, x):
        encoded = []
        for enc in self.encoder:
            x = enc(x)
            encoded.append(x)

        # Last encoder output is not used in any skip_connection:
        _ = encoded.pop()

        for dec in self.decoder:
            enc_output = encoded.pop()
            x = dec(x, enc_output)

        # Return the logits
        return self.logits(x)

    def get_args_summary(self):
        return (f"MaxCH {self.max_channels} - Depth {self.depth} - Layers{self.layers} - "
                f"Kernel {self.kernel_size} - Sampling {self.sampling_method}")

    def get_kwargs(self):
        kwargs = {"nclass": self.nclass, "in_chans": self.in_chans, "max_channels": self.max_channels,
                  "depth": self.depth, "layers": self.layers, "kernel_size": self.kernel_size,
                  "sampling_factor": self.sampling_factor, "sampling_method": self.sampling_method,
                  "skip_connection": self.skip_connection}

        return kwargs
