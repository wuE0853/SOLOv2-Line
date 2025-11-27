import torch.nn as nn
import torch.nn.functional as F
form utils.utils import xavier_init, ConvModule

class FPN(nn.module):
    def __init__(self,
                 in_channels,
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 activation=None):
        """
        in_channels (list[int]): The channels from each layer of backbone,
            such as ResNet50:[256, 512, 1024, 2048]
        out_channels (int): The number of output channels.
        num_outs (int): The number of Pyramids Layer.
            If num_outs > len(in_channels), the extra layer will be generated.
        start_level (int): The start layer of backbone.
            Can skip the front layers by setting a non-zero value.
        end_level (int): The end layer of backbone. -1 stands for the last layer.
        activation (str/None): The activation funtion, like 'relu'.
        """

        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation

        if end_level == -1:
            self.backbone_end_level = self.in_channels[-1]
            assert self.num_outs >= self.num_outs - start_level
        else:
            # if end_level < num_ins, no extra level will be allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=None,
                activation=self.activation
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=None,
                activation=self.activation
                inplace=False
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
