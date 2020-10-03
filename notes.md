# conda environment

conda create -n pyProj
source activate pyProj
conda install -c pytorch pytorch
conda install -c pytorch torchvision
conda install numpy 
conda install matplotlib
conda install -c conda-forge ipdb


# Tutorials

Tutorial handwritten digit recognition using PyTorch and MNIST dataset
- https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

Pytorch and CNN tutorials 
- https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
- https://www.kaggle.com/sdelecourt/cnn-with-pytorch-for-mnist
- https://medium.com/swlh/pytorch-real-step-by-step-implementation-of-cnn-on-mnist-304b7140605a


# Questions

How to evaluate the model?
- only accuracy does not make sense because classes are biased
- what does 70 or 80% accuracy in Wang et al. 2016 actually mean?
- what is the 'simple'/random model to compare to? 

Which loss function and regularization?
- what loss function do Wang et al. use and how to implement it in pytorch (optimizer)
- does the loss function account for the class bias?
- how to implement regularization in pytorch? 

How to handle the class bias?
- over sampling of rare classes does not make sense?
- tailor loss function to penalize rare classes more?

Architecture
- what is the CNN architecture of Wang et al.? 
  > we set window size to 11 and use 5 hidden layers, each with 100 different neurons.

  What does number of neurons mean? Filter number or filter number * window size?
- Do they have different architectures for Q3 and Q8?

How to train the model?
- avoid getting stuck in local optima
- how many epochs and what batch size