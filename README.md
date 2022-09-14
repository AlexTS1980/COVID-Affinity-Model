## Update from 14/09/22: Published in Applied Soft Computing, 116-108261, February 2022, DOI https://doi.org/10.1016/j.asoc.2021.108261
## Update from 03/12/21: To appear in Applied Soft Computing

# COVID-Affinity-Model
BibTex:
```
@article{ter2020one,
  title={One Shot Model For The Prediction of COVID-19 and Lesions 
  Segmentation In Chest CT Scans Through The Affinity Among Lesion Mask Features},
  author={Ter-Sarkisov, Aram},
  journal={medRxiv},
  pages={2020--12},
  year={2020},
  publisher={Cold Spring Harbor Laboratory Press}
}
```
### Affinity model
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-Affinity-Model/blob/master/figures/ssd_affinity.png" width="550" height="200" align="center"/>
</p>

### Region of Interest layer - Lesion Mask Features  

<p align="center">
<img src="https://github.com/AlexTS1980/COVID-Affinity-Model/blob/master/figures/masks.png" width="350" height="200" align="center"/>
</p>  

### Affinity matrix for a COVID-19 image. Left column: after 1 epoch, right column: after 100 epochs

<p align="center">
<img src="https://github.com/AlexTS1980/COVID-Affinity-Model/blob/master/figures/x_feats_ncp.png" width="350" height="150" align="center"/>
</p>

# Segmentation Results (CNCB-NCOV Segmentation Dataset, (http://ncov-ai.big.ac.cn)

|  \# Affinities	| AP@0.5 	| AP@0.75 	| mAP@[0.5:0.95:0.05] 	| 
|:-:	|:-:	|:-:	|:-:|
|  32	| **0.614** 	| 0.382 	| 0.395 	| 
| 64 | 0.603 	| **0.414** 	| **0.422** 	|
|128 | 0.569 	| 0.350 	|0.385|
| 256 |  0.560| 0.347|0.386|
| 512 |  0.548| 0.343|0.386|

# Classification Results (CNCB-NCOV Classification Dataset, (http://ncov-ai.big.ac.cn)

|  \# Affinities	| COVID-19 	| CP 	| Normal 	| F1 score|
|:-:	|:-:	|:-:	|:-:|:-:|
|  32	| 89.39%	|80.25%|98.96% 	|90.30% |
| 64 | 90.68% 	|83.60% 	|97.15% |91.00% 	|
|128 | 86.91% 	| **95.65%** 	|95.45%|**93.80%**|
| 256 | **91.74%**|85.35% |97.26%|91.94%|
| 512 | 90.27% |84.53%| **99.41%**|92.34%|

# Classification Results (iCTCF-CT Classification Dataset, (http://ictcf.biocuckoo.cn)

|  \# Affinities	| COVID-19 	| Normal 	| F1 score|
|:-:	|:-:	|:-:	|:-:|
|  32	| **92.11%**	|80.31%	|83.67% |
| 64 | 86.73%	|**94.20%** |**92.00%** 	|
|128 | 88.88%	|83.85%|85.27%|
| 256 | 77.41%|93.33%|88.78%|
| 512 | 90.49% |89.96%|90.11%|
## Data 
CNCB-NCOV data: (ncov-ai.big.ac.cn/download) with COVIDx-CT [splits](https://github.com/haydengunraj/COVIDNet-CT/blob/master/docs/dataset.md).

iCTCF-CT data: (http://ictcf.biocuckoo.cn/HUST-19.php). Download the nCT(no disease) data. Train and test splits are provided in this repository. I changed the image names to match the convention used in COVIDx-CT: `0` for Negative and `2` for COVID-19.  

## Testing/Evaluating The Model
To test the model trained on CNCB-NCOV datasets:
```
python3.5 test_classification_branch.py --ckpt pretrained_models/affinity_model_128.pth --test_data cncb_ncov/test --device cuda --affinity 128
```
To test the model trained on iCTCF-CT dataset:
```
python3.5 test_classification_branch.py --ckpt pretrained_models/affinity_model_ictcf_64.pth --test_data ictcf/test --device cuda --affinity 64
```
This outputs confusion matrix and F1 score above. Links to models are in `pretrained_models` directory. 
