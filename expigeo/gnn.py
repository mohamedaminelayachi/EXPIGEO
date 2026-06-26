import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from torch_geometric.nn import GATv2Conv


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, 
                 class_to_optimize=1):
        """
        :param class_to_optimize: the target class to optimize,
          aneurysm (label 1) in our case.
        """
        super().__init__()
        self.class_to_optimize = class_to_optimize

    @staticmethod
    def lovasz_grad(gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    
    def lovasz_softmax_flat(self, probs, 
                            labels, 
                            class_to_optimize=1):
        fg = (labels == class_to_optimize).float()
        if fg.sum() == 0:
            return probs.sum() * 0.0

        class_probs = probs[:, class_to_optimize]
        errors = (fg - class_probs).abs()

        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]

        grad = self.lovasz_grad(fg_sorted)
        loss = torch.dot(errors_sorted, grad)

        return loss

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        return self.lovasz_softmax_flat(probs, labels, self.class_to_optimize)


class ResidualBlock(nn.Module):
  def __init__(self, in_channels,
               out_channels,
               heads: int=2,
               concat: bool=False,
               self_loop: bool=True,
               non_linearity: str='gelu',
               dropout: float=0.4):
    """
    :param in_channels: number of input channels.
    :param out_channels: number of output channels.
    :param heads: number of attention heads 
                  (used for GATv2Conv and GATConv)
    :param concat: controls whether the convolution 
                    layer uses concatenation or not.
    :param self_loop: controls whether self-loops is 
                      used or not.
    :param non_linearity: specifies the type of linearity
                          to use (GELU, ReLU).
    :param dropout: the amount of dropout to use.
    """
    super().__init__()

    self.conv = GATv2Conv(in_channels,
                        out_channels,
                        add_self_loops=self_loop,
                        heads=heads,
                        concat=concat)

    self.norm = nn.BatchNorm1d(out_channels)

    if non_linearity.lower() == 'relu':
      self.non_linearity = nn.ReLU()
    elif non_linearity.lower() == 'gelu':
      self.non_linearity = nn.GELU()

    if in_channels != out_channels:
      self.res_proj= nn.Linear(in_channels, out_channels)
    else:
      self.res_proj = nn.Identity()

    self.dropout = nn.Dropout(dropout)

  def forward(self, x, edge_index):

    res = self.res_proj(x)
    out = self.norm(self.conv(x, edge_index))
    out = self.non_linearity(out)
    out = self.dropout(out)
    out = out + res

    return out
  

@dataclass
class ModelConfig:
   
    in_channels: int
    out_channels: int
    num_blocks: int
    blocks_out_channels: list
    num_classes: int
    dropout: float

    def serialize_config_as_yaml(self, save_path: str):
        data = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'num_blocks': self.num_blocks,
            'blocks_out_channels': self.blocks_out_channels,
            'num_classes': self.num_classes,
            'dropout': self.dropout
        }

        with open(save_path, "w") as file:
            yaml.dump(data, file)

    def load_config_from_yaml(self, load_path: str):
      with open(load_path, "r") as file:
          config = yaml.safe_load(file)

      self.in_channels = config['in_channels'] if 'in_channels' in config else 9
      self.out_channels = config['out_channels'] if 'out_channels' in config else 256
      self.num_blocks = config['num_blocks'] if 'num_blocks' in config else 3
      self.blocks_out_channels = (config['blocks_out_channels'] 
                                  if 'blocks_out_channels' in config else [256, 256, 256])
      self.num_classes = config['num_classes'] if 'num_classes' in config else 2
      self.dropout = config['dropout'] if 'dropout' in config else 0.5

      return self



class ExpigeoGNN(nn.Module):
    def __init__(self, in_channels=9,
                 out_channels=256,
                 num_blocks=3,
                 blocks_out_channels=[256, 256, 256],
                 num_classes=2,
                 dropout=0.5,
                 model_config: ModelConfig=None):
        
        """
        :param in_channels: number of input channels.
        :param out_channels: the size of the last graph convolution layer.
        :param num_blocks: number of residual blocks to use.
        :param blocks_out_channels: a list of output sizes of the 
                                    used residual blocks. The list 
                                    size must be the same as the 
                                    given num_blocks.
        :param num_classes: number of classes (2 by default
                            aneurysm vs healthy artery).
        :param dropout: the amount of dropout to use.
        """
        super().__init__()

        if model_config:
           in_channels = model_config.in_channels
           out_channels = model_config.out_channels
           num_blocks = model_config.num_blocks
           blocks_out_channels = model_config.blocks_out_channels
           num_classes = model_config.num_classes
           dropout = model_config.dropout

        assert num_blocks == len(blocks_out_channels), "Residual Blocks Mismatch"

        self.res_blocks = nn.ModuleList()

        self.res_blocks.append(ResidualBlock(in_channels,
                                             blocks_out_channels[0]))

        for i in range(num_blocks - 1):
            self.res_blocks.append(ResidualBlock(blocks_out_channels[i],
                                            blocks_out_channels[i+1]))


        self.f_conv = GATv2Conv(blocks_out_channels[-1], out_channels,
                                add_self_loops=True)

        if in_channels == out_channels:
          self.res_proj = nn.Identity()
        else:
          self.res_proj = nn.Linear(in_channels, blocks_out_channels[-1])

        self.dropout = dropout

        self.projection_head = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(out_channels // 2, out_channels // 4),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(out_channels // 4, num_classes)
        )

        self.num_params = sum(param.numel() for param in self.parameters())

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index

        res = self.res_proj(x)

        for block in self.res_blocks:
            x = block(x, edge_index)

        x = x + res

        x = self.f_conv(x, edge_index)

        x = self.projection_head(x)

        return x