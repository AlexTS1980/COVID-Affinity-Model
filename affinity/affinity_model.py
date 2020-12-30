# Alex Ter-Sarkisov@City, University of London
# Dec 2020: Merry Christmas&A Happy New Year!!!!
# use this method as an extension to Mask R-CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from .faster_rcnn import TwoMLPHead, FastRCNNPredictor
from .mask_rcnn import MaskRCNN, MaskRCNNHeads, MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from .backbone_utils_adjusted import resnet_fpn_backbone
from collections import OrderedDict
from torchvision.ops import misc as misc_nn_ops
from torch.nn.modules.loss import _Loss
from collections import OrderedDict

__all__ = [
    "get_affinity_model"
]

# Main affinity class
class AffinityModel(MaskRCNN):
    def __init__(self, backbone, num_classes=2, 
                 # Faster and Mask R-CNN
                 min_size=512, max_size=512,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=400, rpn_pre_nms_top_n_test=400,
                 rpn_post_nms_top_n_train=200, rpn_post_nms_top_n_test=200,
                 rpn_nms_thresh=0.75,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.75,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=256, box_positive_fraction=0.75,
                 bbox_reg_weights=None,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None,
                 # Alex - SSM
                 box_score_thresh_classifier=-0.01, box_nms_thresh_classifier=0.25, box_detections_per_img_s2new=8,
                 # Alex - Mask+Box Features extractor,
                 box_pool_s2=None, box_head_s2=None, box_predictor_s2=None,
                 mask_pool_s2=None, mask_head_s2=None, mask_predictor_s2=None,
                 # Alex - Affinity model
                 x_stages=3, num_classes_img=3, sieve_layer=None, s2classifier=None, num_affinities=256, affinity=None, s2new_classifier=None, **kwargs):

        out_channels = backbone.out_channels
        # Mask features branch

        # Classification branch
        if box_pool_s2 is None:
            box_pool_s2 = MultiScaleRoIAlign(
                # single feature map
                featmap_names=['0'],
                output_size=7,
                sampling_ratio=2)

        if box_head_s2 is None:
            resolution = box_pool_s2.output_size[0]
            representation_size = 128
            box_head_s2 = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor_s2 is None:
            representation_size = 128
            box_predictor_s2 = FastRCNNPredictor(
                representation_size,
                num_classes)

        if mask_pool_s2 is None:
            mask_pool_s2 = MultiScaleRoIAlign(
                #Alex: the key of the feature map
                featmap_names=['0'],
                output_size=14,
                sampling_ratio=2)

        if mask_head_s2 is None:
            mask_layers = (out_channels,)
            mask_dilation = 1
            mask_head_s2 = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        # add mask predictor: upsample+bn+relu
        if mask_predictor_s2 is None:
            in_channels = mask_head_s2[-2].out_channels
            out_channels = in_channels
            mask_predictor_s2 = MaskRCNNPredictorTruncated(in_channels, out_channels, mask_dilation)

        # Affinity layer, 
        num_feature_maps = mask_predictor_s2.conv_reduce.out_channels
        num_reduce_feature_maps = int(num_feature_maps/2)
        if sieve_layer is None: 
           sieve_layer = MaskFeaturesSieve(num_feature_maps=num_feature_maps, num_reduce_feature_maps=num_reduce_feature_maps, h=28, w=28, apply_linearity=False, final=False) 
           affinity_layer = AffinityLayer(sieve_layer, affinity_matrix_size = box_detections_per_img_s2new, x_stages=x_stages, num_features=num_feature_maps, num_affinities=num_affinities)

        # Image classification batch
        if s2classifier is None:
           s2classifier = ImageClassificationLayerFromMaskFeatures(affinity_feature_size=num_feature_maps, num_classes_img=num_classes_img)
        # instantiate Mask R-CNN:
        # affinity and image classificiaotn module will be passed to the Generalized RCNN
        kwargs.update(affinity=affinity_layer, s2new_classifier=s2classifier)
        super(AffinityModel, self).__init__(backbone, num_classes,
                 # transform parameters
                 min_size, max_size,
                 image_mean, image_std,
                 # RPN parameters
                 rpn_anchor_generator, rpn_head,
                 rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
                 rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
                 rpn_nms_thresh,
                 rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                 rpn_batch_size_per_image, rpn_positive_fraction,
                 # Box parameters
                 box_roi_pool, box_head, box_predictor,
                 box_score_thresh, box_nms_thresh, box_detections_per_img,
                 box_fg_iou_thresh, box_bg_iou_thresh,
                 box_batch_size_per_image, box_positive_fraction,
                 bbox_reg_weights,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None, **kwargs)
        # Alex - SSM
        #
        self.roi_heads.score_thresh_classifier=box_score_thresh_classifier
        self.roi_heads.nms_thresh_classifier=box_nms_thresh_classifier
        self.roi_heads.detections_per_img_s2new = box_detections_per_img_s2new
        #
        # 
        self.roi_heads.box_pool_s2=box_pool_s2
        self.roi_heads.box_head_s2=box_head_s2
        self.roi_heads.box_predictor_s2=box_predictor_s2
        #
        # Alex - Mask Features extractor,
        self.roi_heads.mask_pool_s2=mask_pool_s2
        self.roi_heads.mask_head_s2=mask_head_s2
        self.roi_heads.mask_predictor_s2=mask_predictor_s2

# Mask Feature Sive - contract and expand feature maps
# batch size (16) is not used here explicitly
class MaskFeaturesSieve(nn.Module):

      def __init__(self, num_feature_maps=128, num_reduce_feature_maps=64, h=28, w=28, out_linear_features = None, final=False, apply_linearity=False):
          super(MaskFeaturesSieve, self).__init__()
          # simple block for 'sieving' the features
          # halve the size
          self.conv_down = nn.Conv2d(in_channels=num_feature_maps, out_channels=num_reduce_feature_maps, kernel_size=(2,2), stride=2, padding=0)
          self.bn = nn.BatchNorm2d(num_features=num_reduce_feature_maps)
          # double the size
          self.conv_up = nn.ConvTranspose2d(in_channels=num_reduce_feature_maps, out_channels=num_feature_maps, kernel_size=(2,2), stride=2, padding=0)
          self.relu = nn.ReLU(inplace=False)
          #self.sieve_block = nn.Sequential(*[self.conv_down, self.batch_norm, self.conv_up])
          self.apply_linearity = apply_linearity
          self.final=final
          if self.final and self.apply_linearity:
             self.feature_output = nn.Linear(num_feature_maps*h*w, out_linear_features)

          for name, param in self.named_parameters():
            if "weight" in name and not 'bn' in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0.01)

      # forward method assumes x is the masks feature map, 256x14x14
      def forward(self, x):
          x = self.conv_down(x)
          x = self.relu(self.bn(x))
          x = self.conv_up(x)

          if self.apply_linearity and final:
              m1, m2, m3 = x.size()[1], x.size()[2], x.size()[3]
              x = x.view(-1, m1*m2*m3)
              x = self.linear_features(x)
          return x
 
# accepts the Bx256x14x14 mask features
# outputs the vector for affinity computation
class AffinityLayer(nn.Module):

      def __init__(self, sieve_layer, affinity_matrix_size=None, x_stages=None, num_features=128, normalize=False, apply_linearity = False, final=False, num_affinities = None, **kwargs):
          super(AffinityLayer, self).__init__()
          self.sieve_stages = x_stages
          self.affinity_matrix_size = affinity_matrix_size
          self.num_features = num_features
          self.sieve = []
          for l in range(self.sieve_stages):
                 s = sieve_layer
                 self.sieve.append(s)
          self.sieve = nn.Sequential(*self.sieve)
          # downsize to Cx1x1
          self.conv_d1 = nn.Conv2d(num_features, num_features, kernel_size=(2,2), stride=2, padding=0)
          self.bn1 = nn.BatchNorm2d(num_features=num_features)
          self.conv_d2 = nn.Conv2d(num_features, num_features, kernel_size=(2,2), stride=2, padding=0)
          self.bn2 = nn.BatchNorm2d(num_features=num_features)
          self.conv_d_final = nn.Conv2d(num_features, num_features, kernel_size=(7,7), stride=1, padding=0)
          self.bn3 = nn.BatchNorm2d(num_features=num_features)
          #self.linear_map_image = nn.Linear(num_features, 1)
          self.normalize=normalize
          self.num_rois = None
          #self.bn1 = nn.BatchNorm2d(num_features=num_features)
          self.batch_to_feat_mat = nn.Parameter(torch.randn(affinity_matrix_size, num_affinities), requires_grad=True)
          #self.bn2 = nn.BatchNorm1d(num_features=num_features)
          self.batch_to_feat_vec = nn.Parameter(torch.randn(num_affinities, 1), requires_grad=True)
          self.pars = nn.ParameterList([self.batch_to_feat_mat, self.batch_to_feat_vec])
          #
          for name, param in self.named_parameters():
            if 'weight'in name and 'bn' not in name and 'batch' not in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0.01)

      # for the image classification
      def extract_mask_feature_image(self,x):
          x = x.view(self.affinity_matrix_size, self.num_features)
          x = F.relu(self.linear_map_image(x), inplace=False)
          x = x.unsqueeze(0)

          return x

      def forward(self, x):
          for s in range(self.sieve_stages):
              x = self.sieve(x)
          x = F.relu(self.conv_d1(x), inplace=False)
          x = self.bn1(x)
          x = F.relu(self.conv_d2(x), inplace=False)
          x = self.bn2(x)
          x = F.relu(self.conv_d_final(x), inplace=False)
          x = self.bn3(x)
          # At this point the feature map should be BxCx1x1,
          # Get the affinity matrix BxB
          #x_mask: BxC for the batch affinity
          #x_img: Bx1 for the image
          #x_img = self.extract_mask_feature_image(x)
          x_feat = x.view(self.affinity_matrix_size, self.num_features)
          # W1t*x_feat
          x_feat = self.batch_to_feat_mat.t().matmul(x_feat)
          # x_featt*W2, get 1xC - feature vector 'summarizing the batch'
          # transpose to keep multiplying the same feature across the whole batch
          # 1xC vector
          x_feat = x_feat.t().matmul(self.batch_to_feat_vec).view(1, -1)
          if self.normalize:
             x_feat = x_sigmoid()
          return x_feat

# 07/12
# accepts the affinity matrix (for now)
class ImageClassificationLayerFromMaskFeatures(nn.Module):
      def __init__(self, affinity_feature_size=None, linear_features = 128, num_classes_img = None):
          super(ImageClassificationLayerFromMaskFeatures,self).__init__()
          self.fc_l1 = nn.Linear(affinity_feature_size, linear_features)
          self.fc_l2 = nn.Linear(linear_features, linear_features)
          self.class_predict_logits = nn.Linear(linear_features, num_classes_img)

      # return
      def forward(self, x):
          #x = x.view(self.affinity_matrix_size)
          x = F.relu(self.fc_l1(x), inplace=False)
          x = F.relu(self.fc_l2(x), inplace=False)
          x = self.class_predict_logits(x)
          return x


class MaskRCNNPredictorTruncated(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictorTruncated, self).__init__(OrderedDict([
            ("conv_mask", misc_nn_ops.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("conv_reduce", nn.Conv2d(dim_reduced, dim_reduced, 3, 1, padding=1)),
            ("relu", nn.ReLU(inplace=False)),]))

        for name, param in self.named_parameters():
            if "weight" in name and not 'bn' in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0.01)


# out_channels: 256
def get_affinity_model(backbone=None, pretrained=False, out_channels=256, **kwargs):
    backbone_model =resnet_fpn_backbone (backbone_name=backbone, pretrained=pretrained, out_ch=out_channels)
    main_model = AffinityModel(backbone_model,**kwargs)
    return main_model


