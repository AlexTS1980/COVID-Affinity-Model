# COVID-Affinity-Model

## Affinity model
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-Affinity-Model/blob/master/figures/ssd_affinity.png" width="950" height="300" align="center"/>
</p>

## Region of Interest layer - Lesion Mask Features  

<p align="center">
<img src="https://github.com/AlexTS1980/COVID-Affinity-Model/blob/master/figures/masks.png" width="950" height="600" align="center"/>
</p>

## Affinity matrix for a COVID-19 image. Left column: after 1 epoch, right column: after 100 epochs

<p align="center">
<img src="https://github.com/AlexTS1980/COVID-Affinity-Model/blob/master/figures/x_feats_ncp.png" width="950" height="500" align="center"/>
</p>

# Segmentation Results (CNCB-NCOV Segmentation Dataset, (http://ncov-ai.big.ac.cn/)

|  \# Affinities	| AP@0.5 	| AP@0.75 	| mAP@[0.5:0.95:0.05] 	| 
|:-:	|:-:	|:-:	|:-:|
|  32	| **0.614** 	| 0.382 	| 0.395 	| 
| 64 | 0.603 	| **0.414** 	| **0.422** 	|
|128 | 0.569 	| 0.350 	|0.385|
| 256 |  0.560| 0.347|0.386|
| 512 |  0.548| 0.343|0.386|
