# WRITTEN BY ALEX TER-SARKISOV
# CITY, UNIVERSITY OF LONDON
# alex.ter-sarkisov@city.ac.uk
#
#
import os
import pickle
import re
import sys
import time
from collections import OrderedDict
import config_affinity as config
import cv2
import datasets.dataset_classification as dataset_classification
import matplotlib.pyplot as plt
import models.affinity
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image as PILImage
from models.affinity.rpn import AnchorGenerator
from models.affinity.affinity_model import *
import utils

def main(config, step):
    start = time.time()
    devices = ['cpu', 'cuda']
    assert config.device in devices
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #
    model_name = None
    ckpt = torch.load(config.ckpt, map_location=device)
    if 'model_name' in ckpt.keys():
        model_name = ckpt['model_name']

    # get the thresholds
    test_data_dir = config.test_data_dir

    if model_name is None:
        model_name = "AffinityModel"
    if config.model_name is not None and config.model_name != model_name:
        print("Using model name from the config.")
        model_name = config.model_name

    # classification dataset interface
    # dataset+dataloader
    dataset_class_pars = {'stage': 'eval', 'data': test_data_dir, 'img_size': (512,512)}
    datapoint_class = dataset_classification.COVID_CT_DATA(**dataset_class_pars)
    dataloader_class_pars = {'shuffle': False, 'batch_size': 1}
    dataloader_class_eval = data.DataLoader(datapoint_class, **dataloader_class_pars)
    # load the weights and create the model
    sizes = ckpt['anchor_generator'].sizes
    aspect_ratios = ckpt['anchor_generator'].aspect_ratios
    anchor_generator = AnchorGenerator(sizes, aspect_ratios)
    print("Anchors: ", anchor_generator)
    print(ckpt.keys())
    affinity = ckpt['affinity']
    print("Affinity:", affinity)
    # create modules
    # keyword arguments
    # box_score_thresh_classifier == -0.01
    # Instantiate AffinityModel
    affinity_args = {'rpn_anchor_generator':anchor_generator, 'num_affinities':affinity, 'out_channels':256}
    affinity_classifier = get_affinity_model(backbone = 'resnet18', pretrained = False, **affinity_args)
    # Load weights
    affinity_classifier.load_state_dict(ckpt['model_weights'])
    # Set to the evaluation mode
    print(affinity_classifier)
    affinity_classifier.eval()
    affinity_classifier.affinity_layer.train()
    affinity_classifier=affinity_classifier.to(device)
    # confusion matrix
    cmatrix, c_sens, overall_acc, f1 = step(affinity_classifier, dataloader_class_eval, device)
    print("Done evaluation for {}".format(model_name))
    end=time.time()
    total_time = end-start
    print("Evaluation time {:.2f} seconds".format(total_time))


# returns confusion matrix, precision and recall derived from it
def main_step(model, dl, device):
    confusion_matrix = torch.zeros(3, 3, dtype=torch.int32).to(device)
    for v, b in enumerate(dl):
        X, y = b
        if device == torch.device('cuda'):
            X, y = X.to(device), y.to(device)           
        image = [X.squeeze_(0)]  # remove the batch dimension
        X = utils.normalize_img(image[0], device=device)
        _, pred_scores, _ = model(image)
        # predicted class scores
        confusion_matrix[torch.nonzero(y.squeeze_(0)>0).item(), pred_scores[0]['final_scores'].argmax().item()] += 1
    print("------------------------------------------")
    print("Confusion Matrix for 3-class problem, a total of {0:d} images:".format(len(dl)))
    print("0: Control, 1: Normal Pneumonia, 2: COVID")
    print(confusion_matrix)
    print("------------------------------------------")
    # confusion matrix
    cm = confusion_matrix.float()
    cm[0, :].div_(cm[0, :].sum())
    cm[1, :].div_(cm[1, :].sum())
    cm[2, :].div_(cm[2, :].sum())
    print("------------------------------------------")
    print("Class Sensitivity:")
    print(cm)
    print("------------------------------------------")
    print('Overall accuracy:')
    oa = confusion_matrix.diag().float().sum().div(confusion_matrix.sum())
    print(oa)
    cm_spec = confusion_matrix.float()
    cm_spec[:, 0].div_(cm_spec[:, 0].sum())
    cm_spec[:, 1].div_(cm_spec[:, 1].sum())
    cm_spec[:, 2].div_(cm_spec[:, 2].sum())
    # Class weights: 0, 1, 2
    cw = torch.tensor([0.45, 0.35, 0.2], dtype=torch.float).to(device)
    print("------------------------------------------")
    print('F1 score:')
    f1_score = 2 * cm.diag().mul(cm_spec.diag()).div(cm.diag() + cm_spec.diag()).dot(cw).item()
    print(f1_score)
    # Confusion matrix, class sensitivity, overall accuracy and F1 score
    return confusion_matrix, cm, oa, f1_score


if __name__ == "__main__":
    config_class = config.get_config_pars_affinity("test_classification")
    main(config_class, main_step)
