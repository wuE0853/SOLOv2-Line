#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pdb
import warnings
import torch.nn as nn
import os
import torch.distributed as dist
import torch
import cv2
import numpy as np

form functools import partial
from collections import OrderedDict

os.makedirs('../weights', exist_ok=True)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight ,val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.weight ,bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    # A weight initaion method, preventing VG and EG(梯度消失和梯度爆炸)
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0, 
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def load_state_dict(module, state_dict, strict=False):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    
    load(module)
    load = None # break load->load refernce cycle

    # ignore "num_batch_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if 'num_batch_tracked' not in key]

    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print(err_msg)
    

def bias_init_with_prob(prior_prob):
    # initialize conv/fc bias value according to giving prior probability
    bias_init = float(-np.log(1 - prior_prob) / prior_prob)
    return bias_init


def build_conv_layer(*args, **kwargs):
    return nn.Conv2d(*args, **kwargs)


norm_cfg = {'BN': ('bn', nn.BatchNorm2d),
            'SyncBN': ('bn', nn.SyncBatchNorm),
            'GN': ('gn', nn.GroupNorm)}

def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg, 'The config of norm layer has wrong type'
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type{}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
    
    assert isinstance(postfix, (int, str)), 'The postfix of norm layer should be int or str'
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
        else:
            assert ' num_groups' in cfg_, 'num_groups is necessary in GN'
            layer = norm_layer(num_channels=num_features, **cfg_)
        
        for param in layer.parameters():
            param.requires_grad = requires_grad
        
        return name, layer


class ConvModule(nn.module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """
    def _init_(self,
               in_channels,
               out_channel,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias='auto',
               norm_cfg=None,
               activation='relu',
               inplace=True,
               order=('conv', 'norm', 'act')):
        super(ConvModule, self)._init_()
        assert norm_cfg is None or isinstance(norm_cfg, dict), 'norm_cfg has wrong type'
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3, 'The conv order is wrong'
        assert set(order) == set(['conv', 'act', 'norm']), 'Unrecognized layer name in ConvModule'

        self.with_norm = norm_cfg is not None
        self.with_activation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings('ConvModule has norm and bias at the same time.')

        #build convolution layer
        self.conv = build_conv_layer(in_channels,
                                     out_channel,
                                     kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=bias)
        
        #export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_norm:
            self.gn = nn.GroupNorm(num_groups=32, num_channels=self.out_channels)
        if self.with_activation:
            self.activate = nn.ReLU(inplace=inplace)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        nonlinearity = 'relu' if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.gn, 1, bias=0)
    
    def forward(self, x):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = self.gn(x)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
        return x
    

def matrix_nms(seg_mask, cate_labels, cate_scores, sigma: float = 2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """

    n_samples = cate_labels.shape[0]
    if sum_masks is None:
        sum_masks = seg_mask.sum((1,2)).float()
    seg_mask = seg_mask.reshape(n_samples, -1).float()
    # inter
    inter_matrix = torch.mm(seg_mask, seg_mask.transpose(1, 0))
    # union
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms, kernel == 'gaussian'
    decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
    compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
    decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)

    # update the score
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))
