
from collections import OrderedDict
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.ops import misc as misc_nn_ops
# Alex: use only the last 
from .._utils import IntermediateLayerGetter
from .backbones import resnet


class BackboneWithFPN(nn.Sequential):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        # truncate the model to ooutput return_layers OrderedDict
        body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        # body = truncated backbone
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            #extra_blocks=LastLevelMaxPool(),
        )
        super(BackboneWithFPN, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))
        # that's out_channels for all other modules in the model
        self.out_channels = out_channels


# out_channels=256
def resnet_fpn_backbone(backbone_name, pretrained, out_ch):
    print(backbone_name)
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=nn.BatchNorm2d)
        #norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # freeze layers
    # 29-09:NO!!!!
    #for name, parameter in backbone.named_parameters():
    #    if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
    #        parameter.requires_grad_(False)
    # only use the last layer for FPN
    if backbone_name=='resnet50':
       return_layers = {'layer3':'0'}
       #return_layers = {'layer1':0, 'layer2':1, 'layer3':2, 'layer4':3}
       #in_channels_stage2 = 256
       in_channels_list = [
           backbone.out_channels
       #   in_channels_stage2,
       #   in_channels_stage2 * 2,
       #   in_channels_stage2 * 4,
       #   in_channels_stage2 * 8,
       ]
    else:
       return_layers = {'layer3':'0'}
       in_channels_list = [backbone.out_channels]
    # 256
    out_channels = out_ch
    #print('backbone', backbone, return_layers, out_channels, in_channels_list)
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
