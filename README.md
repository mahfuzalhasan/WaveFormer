# WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation for Efficient Medical Image Segmentation


Pre-print: [https://arxiv.org/abs/2503.23764](https://arxiv.org/abs/2503.23764)

![Overall Network Architecture](images/fig1.jpg)

![WaveFormer Block Architecture](images/fig2.jpg)

![Results on BraTS2023 Dataset](images/fig3.jpg)

## Contact 
If you have any questions about our project, please feel free to contact us by email at mdmahfuzalhasan@ufl.edu.

## Environment install
Clone this repository and navigate to the root directory of the project.

```bash
git clone https://github.com/mahfuzalhasan/WaveFormer.git

```

## Preprocessing, training, testing, inference, and metrics computation

### Data downloading and Preprocessing

Data is from [https://arxiv.org/abs/2305.17033](https://arxiv.org/abs/2305.17033)

We followed data download link and preprocessing steps from SegMamba repo: [https://github.com/ge-xing/SegMamba](https://github.com/ge-xing/SegMamba) 

### Data split and Config

We have provided our train and validation split inside `data_list/default_split`. Test split is acquired from Segmamba author and provided in `data_list/test_list.pkl` file.

Set necessary path variables in the `config.yaml` file. By default, we have provided the paths and other parameters used in our training setup.


### Training 

We used the pre-processed data from the preprocessing step: **data_dir = "./data/fullres/train"** in `config.yaml`


```bash 
python 3_train.py
```

The training logs and checkpoints are saved in: **logdir = f"./logs/{model_name}"**. Set **model_name** in the `config.yaml` file

### Inference 
Once the model is trained, run inference with the best model on the test set. Best model should be in **best_model_path = f"./logs/{model_name}/best_model.pth"**

```bash 
python 4_predict.py
```

The prediction cases will be saved in
**save_path = "./prediction_results/{model_name}"** folder.

### Metrics computation
Run the following script to obtain the Dice score and HD95 on each segmentation target (WT, TC, ET for BraTS2023 dataset)

```bash
python 5_compute_metrics.py
```

## Acknowledgement
Many thanks for these repos for their great contribution! Special thanks goes to the author of Segmamba Papers to provide us with the necessary data and split. Thank you to the National Research Platform ([NRP](https://portal.nrp.ai/)) for providing us with necessary computation resources.

[https://github.com/ge-xing/SegMamba](https://github.com/ge-xing/SegMamba)

[https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI)

