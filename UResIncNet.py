import torch
from torch import nn
from torch.nn import functional as F


def init_weights(module):
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        torch.nn.init.kaiming_normal_(module.weight)
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        torch.nn.init.kaiming_normal_(module.weight)
        torch.nn.init.zeros_(module.bias)


class DilatedResidualInceptionBlock(nn.Module):
    # DilatedResidualBlock from RespNet:
    # Original ResNet backbone for UNET: https://github.com/rishikksh20/ResUnet/blob/master/core/modules.py

    def __init__(self, in_chans, kernel_size=3, neg_slope=0.2):
        super(DilatedResidualInceptionBlock, self).__init__()
        self.in_chans = in_chans
        self.kernel_size = kernel_size
        self.padding = "same"
        self.neg_slope = neg_slope
        # We need to reduce filters/channels by four because then we will concat the filters
        out_chans = in_chans // 4
        self.dilated_conv_block1 = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, padding=self.padding),
            nn.BatchNorm1d(out_chans),
            nn.LeakyReLU(neg_slope),
            nn.Conv1d(out_chans, out_chans, kernel_size=kernel_size, padding=self.padding, dilation=8),
            nn.BatchNorm1d(out_chans)
        )
        self.dilated_conv_block2 = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, padding=self.padding),
            nn.BatchNorm1d(out_chans),
            nn.LeakyReLU(neg_slope),
            nn.Conv1d(out_chans, out_chans, kernel_size=kernel_size, padding=self.padding, dilation=4),
            nn.BatchNorm1d(out_chans)
        )
        self.dilated_conv_block3 = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, padding=self.padding),
            nn.BatchNorm1d(out_chans),
            nn.LeakyReLU(neg_slope),
            nn.Conv1d(out_chans, out_chans, kernel_size=kernel_size, padding=self.padding, dilation=2),
            nn.BatchNorm1d(out_chans)
        )
        self.simple_conv_block = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, kernel_size=1, padding=self.padding),
            nn.BatchNorm1d(out_chans)
        )

    def forward(self, x):
        x1 = self.dilated_conv_block1(x)
        x2 = self.dilated_conv_block2(x)
        x3 = self.dilated_conv_block3(x)
        x4 = self.simple_conv_block(x)

        # Concatenate the channels of the four branches:
        # Each branch has in_channels//4 channels => their sum has in_channels, preserving the number of filters
        x_conv_cat = torch.cat((x1, x2, x3, x4), dim=1)

        # Feature map addition:
        return F.leaky_relu(x_conv_cat + x, self.neg_slope)


class ConvolutionBlock(nn.Module):
    # Consists of Conv -> BatchNorm -> LeakyReLU(neg_slope)
    def __init__(self, in_chans, out_chans, kernel_size=4, sampling_factor=2, sampling_method="conv_stride",
                 dropout: float = 0.0, neg_slope=0.2):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        self.sampling_method = sampling_method
        self.neg_slope = neg_slope

        if sampling_method == "conv_stride":
            stride = sampling_factor
        else:
            stride = 1

        self.block_list = nn.ModuleList(
            [nn.Conv1d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding="same"),
             nn.BatchNorm1d(out_chans),
             nn.LeakyReLU(neg_slope)]
        )

        if dropout > 0.0:
            self.block_list.append(nn.Dropout1d(p=dropout))
        if sampling_method == "pooling":
            self.block_list.append(nn.MaxPool1d(sampling_factor))

    def forward(self, x):
        for f in self.block_list:
            x = f(x)
        return x


class TransposeConvolutionBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, sampling_factor=2):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        # # Transpose convolution:
        # # Lout = (Lin-1)*stride + dilation*(kernel-1) - 2*padding + output_padding + 1
        # # if dilation = 1 then:
        # # Lout = Lin*stride + kernel - stride - 2*padding + output_padding
        # # So we control Lout by stride, and we want the "kernel - stride - 2*padding + output_padding" to be zero
        # # In order to achieve this we want to set padding and output padding in a way that offsets (kernel-stride).
        # # For example if stride == kernel then we set padding=output_padding =0
        #
        # if kernel_size == sampling_factor:
        #     pad = 0
        #     out_pad = 0
        # elif (kernel_size - sampling_factor) % 2 == 0:
        #     pad = 0
        #     out_pad = 0
        # else:
        #     pad = (kernel_size - sampling_factor) // 2
        #     out_pad = (kernel_size - sampling_factor) % 2

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

        self.f = nn.ConvTranspose1d(self.in_chans, self.out_chans,
                                    stride=sampling_factor, kernel_size=kernel_size,
                                    padding=pad, output_padding=out_pad)

    def forward(self, x):
        return self.f(x)


class EncoderBlock(nn.Module):
    # Consists of Conv -> LeakyReLU(neg_slope) -> MaxPool
    def __init__(self, in_chans, out_chans, kernel_size=3, layers=1, sampling_factor=2, sampling_method="conv_stride",
                 dropout: float = 0.0, neg_slope=0.2):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        self.sampling_method = sampling_method
        self.dropout = dropout
        self.neg_slope = neg_slope

        self.encoder = nn.ModuleList()

        if sampling_method == "conv_stride":
            stride = sampling_factor
            diff = (kernel_size - sampling_factor)
            if diff % 2 == 0:
                ks = kernel_size
                pad = diff // 2
            else:
                ks = kernel_size + 1
                pad = (ks - sampling_factor) // 2
        else:
            stride = 1
            ks = kernel_size
            pad = "same"
            self.encoder.append(nn.MaxPool1d(sampling_factor))

        # Simple convolution for channel adjustment and for downscaling (if not max pooling):
        self.encoder.append(nn.Conv1d(in_chans, out_chans, ks, stride=stride, padding=pad))
        self.encoder.append(nn.BatchNorm1d(out_chans))
        self.encoder.append(nn.LeakyReLU(neg_slope))

        for _ in range(layers):
            # Dilated Residual Inception block:
            self.encoder.append(DilatedResidualInceptionBlock(out_chans, kernel_size=kernel_size, neg_slope=neg_slope))

        # Dropout:
        if dropout > 0.0:
            self.encoder.append(nn.Dropout1d(p=dropout))

    def forward(self, x):
        for enc in self.encoder:
            x = enc(x)
        return x


class AttnGatingBlock(nn.Module):
    def __init__(self, x_chans, g_chans, neg_slope=0.2):
        super().__init__()
        self.x_chans = x_chans
        self.g_chans = g_chans
        self.neg_slope = neg_slope

        inter_chans = max(1, min(x_chans, g_chans) // 2)
        self.inter_chans = inter_chans

        # Normally x.shape==g.shape, as g is already up-sampled from a level lower and x is skip connection

        self.phi_g = nn.Sequential(nn.Conv1d(in_channels=g_chans, out_channels=inter_chans, kernel_size=1, stride=1),
                                   nn.BatchNorm1d(inter_chans))
        self.theta_x = nn.Sequential(nn.Conv1d(in_channels=x_chans, out_channels=inter_chans, kernel_size=1, stride=1),
                                     nn.BatchNorm1d(inter_chans))

        self.psi = nn.Sequential(nn.Conv1d(in_channels=inter_chans, out_channels=1, kernel_size=1),
                                 nn.BatchNorm1d(1),
                                 nn.Sigmoid())
        self.relu = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)

    def forward(self, x, g):
        assert x.shape == g.shape
        x1 = self.theta_x(x)
        g1 = self.phi_g(g)
        psi = self.psi(self.relu(x1 + g1))
        out = x * psi
        return out


class DecoderBlock(nn.Module):
    # Consists of transposed convolution -> Conv -> LeakyReLU(neg_slope)
    def __init__(self, in_chans, out_chans, kernel_size=3, layers=1, sampling_factor=2, dropout=0.0,
                 skip_connection=True, expected_skip_connection_chans=None, attention=False, neg_slope=0.2):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.sampling_factor = sampling_factor
        self.dropout = dropout
        self.skip_connection = skip_connection
        self.attention = attention
        self.neg_slope = neg_slope

        if skip_connection:
            if expected_skip_connection_chans is None:
                skip_factor = 1
                # If max_channels was not reached during encoding then: skip_conn_chans == in_chans // 2
                # in_chans // 2 + in_chans // 2 = in_chans
                chans_after_connection = in_chans
                expected_skip_connection_chans = in_chans // 2
            else:
                # skip_conn_chans + in_chans // 2 = in_chans (should be equal in most cases)
                chans_after_connection = expected_skip_connection_chans + in_chans // 2
        else:
            # Channels from transpose convolution:
            chans_after_connection = in_chans // 2

        self.decoder = nn.ModuleList()
        # Up-sampling:
        self.upsample = TransposeConvolutionBlock(in_chans, in_chans // 2, kernel_size=kernel_size,
                                                  sampling_factor=sampling_factor)
        if attention:
            self.att = AttnGatingBlock(x_chans=expected_skip_connection_chans, g_chans=in_chans // 2,
                                       neg_slope=neg_slope)

        # Simple convolution for channel adjustment:
        self.decoder.append(
            nn.Conv1d(chans_after_connection, out_chans, kernel_size=kernel_size, stride=1, padding="same"))
        self.decoder.append(nn.BatchNorm1d(out_chans))
        self.decoder.append(nn.LeakyReLU(neg_slope))

        for _ in range(layers):
            # Dilated Residual Inception block:
            self.decoder.append(DilatedResidualInceptionBlock(out_chans, kernel_size=kernel_size, neg_slope=neg_slope))

        # Dropout:
        if dropout > 0.0:
            self.decoder.append(nn.Dropout1d(p=dropout))

    def forward(self, x, enc_features=None):
        x = self.upsample(x)

        if self.attention:
            enc_features = self.att(x=enc_features, g=x)

        if self.skip_connection:
            x = torch.cat((enc_features, x), dim=1)

        for dec in self.decoder:
            x = dec(x)
        return x


class UResIncNet(nn.Module):
    def __init__(self, nclass=5, in_chans=1, first_out_chans=4, max_channels=512, depth=8, kernel_size=4, layers=1,
                 sampling_factor=2, sampling_method="conv_stride", dropout=0.0, skip_connection=True, attention=False,
                 extra_final_conv=False,
                 custom_weight_init=False, neg_slope=0.2):
        super().__init__()
        if extra_final_conv and first_out_chans == 4:
            # For backwards compatibility, before the existence of first_out_chans
            first_out_chans = 8

        self.nclass = nclass
        self.in_chans = in_chans
        self.first_out_chans = first_out_chans
        self.max_channels = max_channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.layers = layers
        self.sampling_factor = sampling_factor
        self.sampling_method = sampling_method
        self.dropout = dropout
        self.skip_connection = skip_connection
        self.attention = attention
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.extra_final_conv = extra_final_conv
        self.neg_slope = neg_slope

        # first_out_chans = max_channels // (2 ** (depth - 1))
        # if first_out_chans % 4 == 0:
        #     out_chans = first_out_chans
        # else:
        #     out_chans = 4

        out_chans = self.first_out_chans

        # First block is special, input channel dimensionality has to be adjusted otherwise residual block will fail.
        # Also, the first block should not do any down-sampling (stride = 1):
        self.encoder.append(EncoderBlock(self.in_chans, out_chans,
                                         kernel_size=self.kernel_size,
                                         layers=self.layers,
                                         sampling_factor=1,
                                         sampling_method=self.sampling_method,
                                         neg_slope=neg_slope))
        n_same_channel_blocks = 0
        for _ in range(depth - 1):
            if out_chans * 2 <= self.max_channels:
                in_chans, out_chans = out_chans, out_chans * 2
            else:
                in_chans, out_chans = out_chans, out_chans
                n_same_channel_blocks += 1
            self.encoder.append(EncoderBlock(in_chans, out_chans,
                                             kernel_size=self.kernel_size,
                                             layers=self.layers,
                                             sampling_factor=self.sampling_factor,
                                             sampling_method=self.sampling_method,
                                             dropout=self.dropout,
                                             neg_slope=neg_slope))

        for _ in range(depth - 1):

            if out_chans // 2 >= self.first_out_chans and n_same_channel_blocks == 0:
                in_chans, out_chans = out_chans, out_chans // 2
                expected_skip_conn_chans = in_chans // 2
            else:
                in_chans, out_chans = out_chans, out_chans
                n_same_channel_blocks -= 1
                expected_skip_conn_chans = self.max_channels

            self.decoder.append(DecoderBlock(in_chans, out_chans,
                                             kernel_size=self.kernel_size,
                                             layers=self.layers,
                                             skip_connection=self.skip_connection,
                                             expected_skip_connection_chans=expected_skip_conn_chans,
                                             sampling_factor=self.sampling_factor,
                                             attention=attention,
                                             dropout=self.dropout,
                                             neg_slope=neg_slope))

        if self.extra_final_conv:
            self.final_conv_block = nn.Sequential(EncoderBlock(out_chans, out_chans,
                                                               kernel_size=self.kernel_size,
                                                               layers=self.layers,
                                                               sampling_factor=1,
                                                               dropout=self.dropout,
                                                               neg_slope=neg_slope),
                                                  EncoderBlock(out_chans, out_chans,
                                                               kernel_size=self.kernel_size,
                                                               layers=self.layers,
                                                               sampling_factor=1,
                                                               dropout=self.dropout,
                                                               neg_slope=neg_slope)
                                                  )

        # Add a 1x1 convolution to produce final classes
        self.logits = nn.Conv1d(out_chans, nclass, kernel_size=1, stride=1)

        if custom_weight_init:
            self.apply(init_weights)

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

        if self.extra_final_conv:
            x = self.final_conv_block(x)

        # Return the logits
        return self.logits(x)

    def get_args_summary(self):
        # For backwards compatibility with older class which did not have layers attribute:
        if hasattr(self, "first_out_chans"):
            first_out_chans = self.first_out_chans
        else:
            first_out_chans = 4

        if hasattr(self, "layers"):
            layers = self.layers
        else:
            layers = 1

        if hasattr(self, "dropout"):
            dropout = self.dropout
        else:
            dropout = 0.0

        if hasattr(self, "extra_final_conv"):
            extra_final_conv = self.extra_final_conv
        else:
            extra_final_conv = False

        if hasattr(self, "attention"):
            attention = self.attention
        else:
            attention = False

        if hasattr(self, "neg_slope"):
            neg_slope = self.neg_slope
        else:
            neg_slope = 0.2

        return (
            f"FirstOutChans {first_out_chans} - MaxCH {self.max_channels} - Depth {self.depth} - Kernel {self.kernel_size} "
            f"- Layers {layers} - Sampling {self.sampling_method} - Dropout {dropout} - ExtraFinalConv {extra_final_conv} "
            f"- Attention {attention} - LeakyReLU slope {neg_slope}")

    def get_kwargs(self):
        # For backwards compatibility with older class which did not have layers attribute:
        if hasattr(self, "first_out_chans"):
            first_out_chans = self.first_out_chans
        else:
            first_out_chans = 4

        if hasattr(self, "layers"):
            layers = self.layers
        else:
            layers = 1

        if hasattr(self, "dropout"):
            dropout = self.dropout
        else:
            dropout = 0.0

        if hasattr(self, "extra_final_conv"):
            extra_final_conv = self.extra_final_conv
        else:
            extra_final_conv = False

        if hasattr(self, "attention"):
            attention = self.attention
        else:
            attention = False

        if hasattr(self, "neg_slope"):
            neg_slope = self.neg_slope
        else:
            neg_slope = 0.2

        kwargs = {"nclass": self.nclass, "in_chans": self.in_chans, "first_out_chans": first_out_chans,
                  "max_channels": self.max_channels,
                  "depth": self.depth, "kernel_size": self.kernel_size, "layers": layers,
                  "sampling_factor": self.sampling_factor, "sampling_method": self.sampling_method,
                  "skip_connection": self.skip_connection, "dropout": dropout,
                  "extra_final_conv": extra_final_conv, "attention": attention, "neg_slope": neg_slope}
        return kwargs


class ResIncNet(nn.Module):

    def __init__(self, nclass=1, in_size=512, in_chans=1, max_channels=512, depth=8, kernel_size=4, layers=1,
                 sampling_factor=2,
                 sampling_method="conv_stride", skip_connection=True, custom_weight_init=False, neg_slope=0.2):
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
        self.skip_connection = skip_connection
        self.neg_slope = neg_slope

        self.encoder = nn.ModuleList()

        # first_out_chans = max_channels // (2 ** (depth - 1))
        # if first_out_chans % 4 == 0:
        #     out_chans = first_out_chans
        # else:
        #     out_chans = 4

        out_chans = 4

        # First block is special, input channel dimensionality has to be adjusted otherwise residual block will fail.
        # Also, the first block should not do any downsampling (stride = 1):
        self.encoder.append(EncoderBlock(in_chans, out_chans, kernel_size=kernel_size, layers=layers, sampling_factor=1,
                                         sampling_method=sampling_method))

        for _ in range(depth - 1):
            if out_chans * 2 <= max_channels:
                in_chans, out_chans = out_chans, out_chans * 2
            else:
                in_chans, out_chans = out_chans, out_chans

            self.encoder.append(EncoderBlock(in_chans, out_chans, kernel_size=kernel_size, layers=layers,
                                             sampling_factor=sampling_factor, sampling_method=sampling_method))

        out_size = self.in_size // (sampling_factor ** (self.depth - 1))
        assert out_size > 0

        self.fc = nn.Sequential(
            nn.Linear(in_features=out_chans * out_size, out_features=out_chans * out_size),
            nn.BatchNorm1d(out_chans * out_size),
            nn.LeakyReLU(neg_slope),
            nn.Linear(in_features=out_chans * out_size, out_features=out_chans * out_size),
            nn.BatchNorm1d(out_chans * out_size),
            nn.LeakyReLU(neg_slope),
        )

        self.logits = nn.Linear(out_chans * out_size, nclass)

        if custom_weight_init:
            self.apply(init_weights)

    def forward(self, x):
        encoded = []
        for enc in self.encoder:
            x = enc(x)
            encoded.append(x)

        m = nn.Flatten(start_dim=1)
        x = m(x)
        x = self.fc(x)
        # Return the logits
        return self.logits(x)

    def get_args_summary(self):
        # For backwards compatibility with older class which did not have layers attribute:
        if hasattr(self, "layers"):
            layers = self.layers
        else:
            layers = 1

        if hasattr(self, "neg_slope"):
            neg_slope = self.neg_slope
        else:
            neg_slope = 1

        return (f"MaxCH {self.max_channels} - Depth {self.depth} - Kernel {self.kernel_size} "
                f"- Layers {layers} - Sampling {self.sampling_method} - NegSlope {neg_slope}")

    def get_kwargs(self):
        # For backwards compatibility with older class which did not have layers attribute:
        if hasattr(self, "layers"):
            layers = self.layers
        else:
            layers = 1

        if hasattr(self, "neg_slope"):
            neg_slope = self.neg_slope
        else:
            neg_slope = 1

        kwargs = {"nclass": self.nclass, "in_size": self.in_size, "in_chans": self.in_chans,
                  "max_channels": self.max_channels, "depth": self.depth, "kernel_size": self.kernel_size,
                  "layers": layers, "sampling_factor": self.sampling_factor, "sampling_method": self.sampling_method,
                  "skip_connection": self.skip_connection, "neg_slope": neg_slope}
        return kwargs
