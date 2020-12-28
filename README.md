# COVID-Affinity-Model

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
|  32	| 92.11%	|80.31%	|83.67% |
| 64 | 86.73%	|94.20% |92.00% 	|
|128 | 88.88%	|83.85%|85.27%|
| 256 | 77.41%|93.33%|88.78%|
| 512 | 90.49% 89.96%|90.11%|
