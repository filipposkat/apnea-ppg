import torch
from torch import nn
from torch.nn import functional as F


class DilatedResidualInceptionBlock(nn.Module):
    # DilatedResidualBlock from RespNet:
    # Original ResNet backbone for UNET: https://github.com/rishikksh20/ResUnet/blob/master/core/modules.py

    def __init__(self, in_chans, kernel_size=4):
        super(DilatedResidualInceptionBlock, self).__init__()
        self.padding = "same"
        # We need to reduce filters/channels by four because then we will concat the filters
        out_chans = in_chans // 4
        self.dilated_conv_block1 = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, padding=self.padding),
            nn.BatchNorm1d(out_chans),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_chans, out_chans, kernel_size=kernel_size, padding=self.padding, dilation=8),
            nn.BatchNorm1d(out_chans)
        )
        self.dilated_conv_block2 = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, padding=self.padding),
            nn.BatchNorm1d(out_chans),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_chans, out_chans, kernel_size=kernel_size, padding=self.padding, dilation=4),
            nn.BatchNorm1d(out_chans)
        )
        self.dilated_conv_block3 = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, padding=self.padding),
            nn.BatchNorm1d(out_chans),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_chans, out_chans, kernel_size=kernel_size, padding=self.padding, dilation=2),
            nn.BatchNorm1d(out_chans)
        )
        self.simple_conv_block = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, padding=self.padding),
            nn.BatchNorm1d(out_chans)
        )

    def forward(self, x):
        x1 = self.dilated_conv_block1(x)
        x2 = self.dilated_conv_block1(x)
        x3 = self.dilated_conv_block1(x)
        x4 = self.simple_conv_block(x)

        # Concatenate the channels of the four branches:
        # Each branch has in_channels//4 channels => their sum has in_channels, preserving the number of filters
        x_conv_cat = torch.cat((x1, x2, x3, x4), dim=1)

        # Feature map addition:
        return F.leaky_relu(x_conv_cat + x, 0.2)


class ConvolutionBlock(nn.Module):
    # Consists of Conv -> BatchNorm -> LeakyReLU(0.2)
    def __init__(self, in_chans, out_chans, kernel_size=4, sampling_factor=2, sampling_method="conv_stride",
                 dropout: float = 0.0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        self.sampling_method = sampling_method

        if sampling_method is "conv_stride":
            stride = sampling_factor
        else:
            stride = 1

        self.block_list = nn.ModuleList(
            [nn.Conv1d(in_chans, out_chans, kernel_size, stride=stride, padding="same"),
             nn.BatchNorm1d(out_chans),
             nn.LeakyReLU(0.2)]
        )

        if dropout > 0.0:
            self.block_list.append(nn.Dropout1d(p=dropout))
        if sampling_method is "pooling":
            self.block_list.append(nn.MaxPool1d(sampling_factor))

    def forward(self, x):
        for f in self.block_list:
            x = f(x)
        return x


class TransposeConvolutionBlock(nn.Module):
    def __init(self, in_chans, out_chans, kernel_size=3, sampling_factor=2):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

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
            pad = 0
            out_pad = 0
        else:
            pad = (kernel_size - sampling_factor) // 2
            out_pad = (kernel_size - sampling_factor) % 2

        self.f = nn.ConvTranspose1d(in_chans, self.out_chans,
                                    stride=sampling_factor, kernel_size=kernel_size,
                                    padding=pad, output_padding=out_pad)

    def forward(self, x):
        return self.f(x)


class EncoderBlock(nn.Module):
    # Consists of Conv -> LeakyReLU(0.2) -> MaxPool
    def __init__(self, in_chans, out_chans, kernel_size=4, sampling_factor=2, sampling_method="conv_stride",
                 dropout: float = 0.0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        self.sampling_method = sampling_method
        self.dropout = dropout

        self.encoder = nn.ModuleList()

        if sampling_method is "conv_stride":
            stride = sampling_factor
        else:
            stride = 1
            self.encoder.append(nn.MaxPool1d(sampling_factor))

        # Simple convolution for channel adjustment and for downscaling (if not max pooling):
        self.encoder.append(nn.Conv1d(in_chans, out_chans, kernel_size, stride=stride, padding="same"))
        self.encoder.append(nn.BatchNorm1d(out_chans))
        self.encoder.append(nn.LeakyReLU(0.2))

        # Dilated Residual Inception block:
        self.encoder.append(DilatedResidualInceptionBlock(out_chans, kernel_size=kernel_size))

        # Dropout:
        if dropout > 0.0:
            self.encoder.append(nn.Dropout1d(p=dropout))

    def forward(self, x):
        for enc in self.encoder:
            x = enc(x)
        return x


class DecoderBlock(nn.Module):
    # Consists of 2x2 transposed convolution -> Conv -> LeakyReLU(0.2)
    def __init__(self, in_chans, out_chans, kernel_size=4, skip_connection=True, sampling_factor=2,
                 dropout=0.0):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        self.dropout = dropout
        self.skip_connection = skip_connection

        # Up-sampling:
        skip_factor = 1 if skip_connection else 2
        self.decoder = nn.ModuleList()
        self.upsample = TransposeConvolutionBlock(in_chans, in_chans // skip_factor, kernel_size=kernel_size,
                                                  sampling_factor=sampling_factor)

        # Simple convolution for channel adjustment:
        self.decoder.append(
            nn.Conv1d(in_chans // skip_factor, out_chans, kernel_size=kernel_size, stride=1, padding="same"))
        self.decoder.append(nn.BatchNorm1d(out_chans))
        self.decoder.append(nn.LeakyReLU(0.2))

        # Dilated Residual Inception block:
        self.decoder.append(DilatedResidualInceptionBlock(out_chans, kernel_size=kernel_size))

        # Dropout:
        if dropout > 0.0:
            self.decoder.append(nn.Dropout1d(p=dropout))

    def forward(self, x, enc_features=None):
        x = self.tconv(x)
        if self.skip_connection:
            x = torch.cat((enc_features, x), dim=1)
        for dec in self.decoder:
            x = dec(x)
        return x


class UResIncNet(nn.Module):
    def __init__(self, nclass=1, in_chans=1, max_channels=512, depth=8, kernel_size=4, sampling_factor=2,
                 sampling_method="conv_stride", skip_connection=True):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        first_out_chans = max_channels // (2 ** depth)
        if first_out_chans % 4 == 0:
            out_chans = first_out_chans
        else:
            out_chans = 4

        # First block is special, input channel dimensionality has to be adjusted otherwise residual block will fail.
        # Also the first block should not do any downsampling (stride = 1):
        self.encoder.append(EncoderBlock(in_chans, out_chans, kernel_size=kernel_size, sampling_factor=1,
                                         sampling_method=sampling_method))
        for _ in range(depth - 1):
            self.encoder.append(EncoderBlock(in_chans, out_chans, kernel_size=kernel_size,
                                             sampling_factor=sampling_factor, sampling_method=sampling_method))
            if out_chans * 2 <= max_channels:
                in_chans, out_chans = out_chans, out_chans * 2
            else:
                in_chans, out_chans = out_chans, out_chans

        out_chans = in_chans // 2
        for _ in range(depth - 1):
            self.decoder.append(DecoderBlock(in_chans, out_chans, kernel_size=kernel_size,
                                             skip_connection=skip_connection, sampling_factor=sampling_factor))
            if out_chans // 2 >= 4:
                in_chans, out_chans = out_chans, out_chans // 2
            else:
                in_chans, out_chans = out_chans, out_chans

        # Add a 1x1 convolution to produce final classes
        self.logits = nn.Conv1d(in_chans, nclass, 1, 1)

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
