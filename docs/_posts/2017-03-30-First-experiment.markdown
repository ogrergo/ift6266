---
layout: post
title:  "First experiment: a variational auto-encoder"
date:   2017-03-30 12:32:12 -0400
---

# First experiment : an auto-encoder
## The auto-encoder
First, I have trained a simple auto-encoder to predict the center given the border. I coded this model with keras, you can find the code [here](https://github.com/ogrergo/ift6266/blob/master/keras_models/auto_encoder.py).


### Training
I trained the model on (border, full image) pairs with Adam optimizer to minimize the L2 loss.

### Results
The results are not promising, the L2 loss producing blur images. Here some results I get on the training set:

[ae]()
