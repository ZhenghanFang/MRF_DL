import torch
import torch.nn as nn
import functools

class conv_block(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d):
        super(conv_block, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU()
            )
    def forward(self, input):
        return self.model(input)

class PixelShuffle_downscale(nn.Module):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    The inverse of and modified from torch.nn.PixelShuffle
    """
    def __init__(self, downscale_factor):
        super(PixelShuffle_downscale, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        return pixel_shuffle_downscale(input, self.downscale_factor)

    def extra_repr(self):
        return 'downscale_factor={}'.format(self.downscale_factor)


def pixel_shuffle_downscale(input, downscale_factor):
    r"""Rearranges elements in a tensor of shape :math:`[C, H*r, W*r]` to a
    tensor of shape :math:`[*, C*r^2, H, W]`.

    The inverse of and modified from torch.nn.functional.pixel_shuffle
    """
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor
    input_view = input.view(
        batch_size, channels, out_height, downscale_factor, 
        out_width, downscale_factor)
    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    channels *= downscale_factor ** 2
    return shuffle_out.view(batch_size, channels, out_height, out_width)
