# Algorithmic Bias

[![Join the chat at https://gitter.im/KWoC-americast/algo-bias](https://badges.gitter.im/KWoC-americast/algo-bias.svg)](https://gitter.im/KWoC-americast/algo-bias?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

The aim of this project is to classify the income category people in a binary fashion as per their attributes. Next, we try to find out the factors which have contributed to the output of our model, at every layer, using a technique called [Layerwise Relevance Propagation (LRP)](http://heatmapping.org/). Three variants of the implementation have been presented in this project. `main.py` is PyTorch based and `tmain.py` is TensorFlow based. LRP is supported by `kmain.py` and `interprettensor/` which are in Keras (preferably through theano) and interprettonsor TF wrapper respectively. We also generate heatmaps for various layers against the input using the Keras variant.

## Setup

```
pip install theano innvestigate
```

The package `innvestigate` installs a variant of keras which needs a version of tensorflow which is currently unsupported. Hence, we use `theano` for this project. We adjust the `backend` accordingly at `~/.keras/keras.json`.

## Generating heatmaps

Next, we generate heatmaps using

```
python train.py
python lrp_matrix.py
python plot.py
```

Please make sure you run `train.py` in inference mode by answering the questions prompted accordingly.

## Sample heatmap

![](pic_1.png)
