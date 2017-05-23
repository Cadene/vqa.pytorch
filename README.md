# Visual Question Answering in pytorch

This repo was made by [Remi Cadene](http://remicadene.com) and [Hedi Ben-Younes](https://twitter.com/labegne), two PhD Students working on VQA at [UPMC-LIP6](http://lip6.fr). We developped this code in the frame of a research paper called [MUTAN: Multimodal Tucker Fusion for VQA](https://arxiv.org/abs/1705.06676) which is (as far as we know) the current state-of-the-art on the [VQA-1 dataset](http://visualqa.org).

The goal of this repo is two folds:
- to make it easier to reproduce our results,
- to provide an efficient and modular code base to the community for further research on other VQA datasets.

If you have any questions about our code or model, don't hesitate to contact us or to submit any issues. Pull request are welcome!

## Introduction

### What is the task about?

The task is about training models in a end-to-end fashion on a multimodal dataset made of triplets:

- an **image** with no other information than the raw pixels,
- a **question** about visual content(s) on the associated image,
- a short **answer** to the question (one or a few words). 

As you can see in the illustration bellow, two different triplets (but same image) of the VQA dataset are represented. The models need to learn rich multimodal representations to be able to give the right answers.

<p align="center">
    <img src="https://raw.githubusercontent.com/Cadene/vqa.pytorch/master/doc/vqa_task.png" width="600"/>
</p>

The VQA task is still on active research. However, when it will be solved, it could be very useful to improve human-to-machine interfaces (especially for the blinds).

### Quick insight about our method

The VQA community developped an approach based on four learnable components:

- a question model which can be a LSTM, GRU, or pretrained Skipthoughts,
- an image model which can be a pretrained VGG16 or Resnet-152,
- a fusion scheme which can be an element-wise sum, concatenation, [MCB](https://arxiv.org/abs/1606.01847), [MLB](https://arxiv.org/abs/1610.04325), or [Mutan](https://arxiv.org/abs/1705.06676),
- optionally, an attention scheme which may have several "glimpses".

<p align="center">
    <img src="https://raw.githubusercontent.com/Cadene/vqa.pytorch/master/doc/mutan.png" width="400"/>
</p>

One of our claim is that the multimodal fusion between the image and the question representations is a critical component. Thus, our proposed model uses a Tucker Decomposition of the correlation Tensor to model reacher multimodal interactions in order to provide proper answers. Our best model is based on :

- a pretrained Skipthoughts for the question model,
- features from a pretrained Resnet-152 (with images of size 3x448x448) for the image model,
- our proposed Mutan (based on a Tucker Decomposition) for the fusion scheme,
- an attention scheme with two "glimpses".

## Using this code

### Requirements

#### Installation

First install python 3 (we don't provide support for python 2). We advise you to install python 3 and pytorch with Anaconda:

- [python with anaconda](https://www.continuum.io/downloads)
- [pytorch with CUDA](http://pytorch.org)

```
conda create --name vqa python=3
source activate vqa
conda install pytorch torchvision cuda80 -c soumith
```

Then clone the repo (with the `--recursive` flag for submodules) and install the complementary requirements:

```
cd $HOME
git clone --recursive https://github.com/Cadene/vqa.pytorch.git 
cd vqa.pytorch
pip install -r requirements.txt
```

#### Submodules

Our code has two external dependencies:

- [VQA](https://github.com/Cadene/VQA) is used to evaluate results files on the valset with the OpendEnded accuracy,
- [skip-thoughts.torch](https://github.com/Cadene/skip-thoughts.torch) is used to import pretrained GRUs and embeddings.

#### Data

Data will be automaticaly downloaded and preprocessed when needed. Links to data are stored in `vqa/datasets/vqa.py` and `vqa/datasets/coco.py`.


### Reproducing results

#### Features

As we first developped on Lua/Torch7, we used the features of [Resnet-152 pretrained with Torch7](https://github.com/facebook/fb.resnet.torch). We plan to port the model in pytorch as well. Meanwhile, you can download the features as following:

```
mkdir -p data/coco/extract/arch,fbresnet152torch
cd data/coco/extract/arch,fbresnet152torch
wget https://data.lip6.fr/coco/trainset.hdf5
wget https://data.lip6.fr/coco/trainset.txt
wget https://data.lip6.fr/coco/valset.hdf5
wget https://data.lip6.fr/coco/valset.txt
wget https://data.lip6.fr/coco/testset.hdf5
wget https://data.lip6.fr/coco/testset.txt
```

/!\ Notice that we've tried the features of [Resnet-152 pretrained with pytorch](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) and got lower results.

#### Pretrained models

We currently provide three models trained with our old Torch7 code and ported to Pytorch:

- MutanNoAtt trained on the VQA-1 trainset,
- MLBAtt trained on the VQA-1 trainvalset and VisualGenome,
- MutanAtt trained on the VQA-1 trainvalset and VisualGenome.

```
mkdir -p logs/vqa
cd logs/vqa
wget http://webia.lip6.fr/~cadene/Downloads/vqa.pytorch/logs/vqa/mutan_noatt_train.zip 
wget http://webia.lip6.fr/~cadene/Downloads/vqa.pytorch/logs/vqa/mlb_att_trainval.zip 
wget http://webia.lip6.fr/~cadene/Downloads/vqa.pytorch/logs/vqa/mutan_att_trainval.zip 
```

Even if we provide results files associated to our pretrained models, you can evaluate them once again on the valset, testset and testdevset using a single command:

```
python train.py -e --path_opt options/vqa/mutan_noatt_train.yaml --resume ckpt
python train.py -e --path_opt options/vqa/mlb_noatt_trainval.yaml --resume ckpt
python train.py -e --path_opt options/vqa/mutan_att_trainval.yaml --resume ckpt
```

To obtain test and testdev results, you will need to zip your result json file (name it as `results.zip`) and to submit it on the [evaluation server](https://competitions.codalab.org/competitions/6961).

### Documentation

#### Architecture

```
.
├── options        # default options dir containing yaml files
├── logs           # experiments dir containing directories of logs (one by experiment)
├── data           # datasets directories
|   ├── coco       # images and features
|   ├── vqa        # raw, interim and processed data
|   └── ...
├── vqa            # vqa package dir
|   ├── datasets   # datasets classes & functions dir (vqa, coco, images, features, etc.)
|   ├── external   # submodules dir (VQA, skip-thoughts.torch)
|   ├── lib        # misc classes & func dir (engine, logger, dataloader, etc.)
|   └── models     # models classes & func dir (att, fusion, notatt, seq2vec)
|
├── train.py       # train & eval models
├── eval_res.py    # eval results files with OpenEnded metric
├── extract.pt     # extract features from coco with CNNs
└── visu.ipynb     # visualizing logs (under development)
```

#### Options

There are three kind of options:

- options from the yaml options files stored in the `options` directory which are used as default (path to directory, logs, model, features, etc.)
- options from the ArgumentParser in the `train.py` file which are set to None and can overwrite default options (learning rate, batch size, etc.)
- options from the ArgumentParser in the `train.py` file which are set to default values (print frequency, number of threads, resume model, evaluate model, etc.)

You can easly add new options in your custom yaml file if needed. Also, if you want to grid search a parameter, you can add an ArgumentParser option and modify the dictionnary in `train.py:L80`.

#### Datasets

We currently provide three datasets:

- [COCOImages](http://mscoco.org/) currently used to extract features, it comes with three datasets: trainset, valset and testset
- COCOFeatures used by any VQA datasets
- [VQA](http://www.visualqa.org/vqa_v1_download.html) comes with four datasets: trainset, valset, testset (including test-std and test-dev) and "trainvalset" (concatenation of trainset and valset)

We plan to add:

- [VisualGenome](http://visualgenome.org/)
- [VQA2](http://www.visualqa.org/)
- [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/)

#### Models

We currently provide four models:

- MLBNoAtt: a strong baseline (BayesianGRU + Element-wise product)
- [MLBAtt](https://arxiv.org/abs/1610.04325): the previous state-of-the-art which add an attention strategy
- MutanNoAtt: our proof of concept (BayesianGRU + Mutan Fusion)
- MutanAtt: the current state-of-the-art

We plan to add several other strategies in the futur.

### Quick examples

#### Extract features from COCO

The needed images will be automaticaly downloaded to `dir_data` and the features will be extracted with a resnet152 by default.

There are three options for `mode` :

- `att`: features will be of size 2048x14x14,
- `noatt`: features will be of size 2048,
- `both`: default option.

Beware, you will need some space on your SSD:

- 32GB for the images,
- 125GB for the train features,
- 123GB for the test features,
- 61GB for the val features.

```
python extract.py -h
python extract.py --dir_data data/coco --data_split train
python extract.py --dir_data data/coco --data_split val
python extract.py --dir_data data/coco --data_split test
```

Note: By default our code will share computations over all available GPUs. If you want to select only one or a few, use the following prefix:

```
CUDA_VISIBLE_DEVICES=0 python extract.py
CUDA_VISIBLE_DEVICES=1,2 python extract.py
```

#### Train models on VQA

Display help message, selected options and run default. The needed data will be automaticaly downloaded and processed using the options in `options/default.yaml`.

```
python train.py -h
python train.py --help_opt
python train.py
``` 

Run a MutanNoAtt model with default options.

```
python train.py --path_opt options/vqa/mutan_noatt.yaml --dir_logs logs/vqa/mutan_noatt
```

Run a MutanAtt model on the trainset and evaluate on the valset after each epoch.

```
python train.py --vqa_trainsplit train --path_opt options/vqa/mutan_att.yaml 
``` 

Run a MutanAtt model on the trainset and valset (by default) and run throw the testset after each epoch (produce a results file that you can submit to the evaluation server).

```
python train.py --vqa_trainsplit trainval --path_opt options/vqa/mutan_att.yaml
``` 


#### Restart training

Restart the model from the last checkpoint.

```
python train.py --path_opt options/vqa/mutan_noatt.yaml --dir_logs logs/vqa/mutan_noatt --resume ckpt
```

Restart the model from the best checkpoint.

```
python train.py --path_opt options/vqa/mutan_noatt.yaml --dir_logs logs/vqa/mutan_noatt --resume best
```

#### Evaluate models on VQA

Evaluate the model from the best checkpoint. If your model has been trained on the training set only (`vqa_trainsplit=train`), the model will be evaluate on the valset and will run throw the testset. If it was trained on the trainset + valset (`vqa_trainsplit=trainval`), it will not be evaluate on the valset.

```
python train.py --vqa_trainsplit train --path_opt options/vqa/mutan_att.yaml --dir_logs logs/vqa/mutan_att --resume best -e
```

## Acknowledgment

Special thanks to the authors of [MLB](https://arxiv.org/abs/1610.04325) for providing some [Torch7 code](https://github.com/jnhwkim/MulLowBiVQA), [MCB](https://arxiv.org/abs/1606.01847) for providing some [Caffe code](https://github.com/akirafukui/vqa-mcb), and our professors and friends from LIP6 for the perfect working atmosphere.