# asGNN

Adaptive Spatial Graph Neural Network (asGNN) is a computational method to predict spatial gene expression profile from tissue morphorogy such as staining images. Following the framework from the study [1], asGNN applies the smoothing-based variational optimization to search for a spatial graph reflecting actual spatial proximity in gene expression among capturing spots during the training. The learned spatial graph can be further utilized to delineate the spatial domains, facilitating the understanding of the spatial organization within the tissue.

![](https://github.com/song0309/asGNN/blob/main/figures/asGNN_workflow.png)

System and package requirements
--------------------------------------------------------------------------------

The code was developed based on the prevous work available at [Github link](https://github.com/gaoyuanwang1976/GraphPartition_SBGNN) and tested on a cluster with Linux system with 30 CPUs and 256GB memory.

The Python package dependencies are listed in the `gnn_ray.yml`, and the virtual environment can be created by running the following the command:
```
conda env create -f gnn_ray.yml
```

Data preparation
--------------------------------------------------------------------------------

The breast cancer tissue data used in this study were obtained from the works [2] and [3]. We followed the data preproessing in the study [2] to extract convolutional features and applied the model in the study [4] to derive morphological features for each tissue image, and provided the prepocessed data for holdout and external validation under the folders `data/holdout` and `data/external` separately. 

Usage
--------------------------------------------------------------------------------

The asGNN model can be trained by running the following commands:

```
conda activate gnn_ray
python asGNN_holdout.py -d data/holdout/<features>/section_list.dat --graph_path data/holdout/<features> --partition_ratio '13:5:5' -m 'GTN' --corr -b 8 --transform 'offset' --sample_size 30 -o <result-folder>
```

where `<features>` can be replaced with either morphological feature folder `morph` or convolution feature folder `conv`, `<result-folder>` should be specified as the folder where you want to save the models. In the result folder, the hyperparameters for the backbone graph neural network (e.g. # of layers, # of hidden units, or learning rate) and Affinity Propagation (AP) clustering ($\alpha$ and $\beta$) will be saved into `GNN_hyperparams_tuning.npz` and `AP_clustering_hyperparams_tuning.npz`, respectively. The parameters linear meta transformation and GNN models from i-th meta-epoch will be placed under the folder with name `t<i>`.
  
Reference
--------------------------------------------------------------------------------

[1] [A Variational Graph Partitioning Approach to Modeling Protein Liquid-liquid Phase Separation](https://www.biorxiv.org/content/10.1101/2024.01.20.576375v1.full), Gaoyuan Wang, Jonathan H Warrell, Suchen Zheng, Mark Gerstein, bioRxiv (2024).

[2] [Integrating spatial gene expression and breast tumour morphology via deep learning](https://www.nature.com/articles/s41551-020-0578-x), Bryan He, Ludvig Bergenstråhle, Linnea Stenbeck, Abubakar Abid, Alma Andersson, Åke Borg, Jonas Maaskola, Joakim Lundeberg and James Zou, Nature biomedical engineering (2020).

[3] [Spatial deconvolution of HER2-positive breast cancer delineates tumor-associated cell type interactions](https://www.nature.com/articles/s41467-021-26271-2), Alma Andersson, Ludvig Larsson, Linnea Stenbeck, Fredrik Salmén, Anna Ehinger, Sunny Z. Wu, Ghamdan Al-Eryani, Daniel Roden, Alex Swarbrick, Åke Borg, Jonas Frisén, Camilla Engblom and Joakim Lundeberg, Nature communications (2021).

[4] [Automated gastric cancer diagnosis on H&E-stained sections; ltraining a classifier on a large scale with multiple instance machine learning](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/8676/867605/Automated-gastric-cancer-diagnosis-on-H38E-stained-sections-ltraining-a/10.1117/12.2007047.short), Eric Cosatto, Pierre-Francois Laquerre, Christopher Malon, Hans-Peter Graf, Akira Saito, Tomoharu Kiyuna, Atsushi Marugame, and Ken'ichi Kamijo, Medical Imaging 2013: Digital Pathology (2013).

