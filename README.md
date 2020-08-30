# U-Net-Pred

<img src="U-Net-Pred.png" alt="hi" class="inline"/>

The neural network architecture was based on the U-Net architecture (Falk et al., 2019; Ronneberger et al., 2015), with the following modifications:  For initialization, the encoder part of U-Net was replaced by all the convolutional and pooling layers of the VGG-16 network,  using the Keras implementation (Simonyan and Zisserman, 2014; Chollet et al., 2015).  Transfer learning using VGG-16 has been previously used in image segmentation (Iglovikov and Shvets, 2018), image reconstruction (Uhrig et al., 2017), style transfer (Gatys et al., 2015), and image inpainting (Liu et al., 2018).  The resulting network architecture consisted of five blocks, each of which had two or three convolutional layers (3×3) with ReLU (rectified linear) activation functions, followed by a max-pooling (2×2) operation. The decoder consisted of five blocks, each with a nearest-neighbor upsampling layer (2×2), followed by two convolutional layers.  The output layer was a convolutional layer with, as is conventional, a linear activation function.  
All convolution operations in the network,  including  the  VGG-16  network,  were  implemented  as partial convolutions. Partial convolution has been introduced with the sparsity-invariant convolutional network where the input to each convolution is paired with a binary mask indicating which pixels are observable or missing, respectively (Uhriget al., 2017). Partial observability of the inputs during the training makes the network robust to input sparsity, regardless of the task of the network.  We implemented a modified version with mask updates per network operation, as described in (Liu et al., 2018). The idea of partial convolution is that the missing region is gradually filled, and that the filled-in information is used for filling in the rest of the missing pixels in an iterative way.

For more details, please see the methods part of our paper:


[Predictability in natural images determines V1 firing rates and synchronization: A deep neural network approach](https://www.biorxiv.org/content/10.1101/2020.08.10.242958v1) 


# Installation

Clone/Fork the repository to use the scripts.

```shell
git clone https://github.com/uranc/gamma-net.git
```

## Requirements
tensorflow v1.14 (cpu/gpu), keras-contrib
```shell
pip install tensorflow==1.14
or
pip install tensorflow-gpu==1.14
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

# Usage

## Command-Line

You can use the pre-trained model based on VGG-16 to predict gamma peak value in log10 space. Requires the input data to be a numpy array.
```shell
python pred.py --mode predict --model f4 --input examples/sample.npy
```

Requires TFRecords as an input file

```shell
python pred.py --mode train --e save_name
```

## Jupyter notebook

- paper figure 

# To Do List
  - TFRecords documentation / loss function for training
  - cleanup / comment
  
# New Features  
  - Variable Input size
  - Variable Mask
  - directory of images
  - numpy
  
  

