# PSIML - DEPTH ESTIMATION
## About
Following is the project by Anastasja Rakić and Sara Dragutinovic, PSIML 8.

Huge thanks to mentors Stefan Mojsilović and Vukašin Ranković.


The goal of our project is monocular depth estimation using U-NET stucture.
All of the code is in Python and we used PyTorch.
First, we implemented U-NET from scratch, but the results weren't the best.
To improve them, we changed the loss, from usual MSE loss to sum of three losses: MSE, gradient and normal loss.
For even better results, we substituted U-NET encoder with pretrained ResNet50.
## Dataset
All our models were trained on **NYU Depth V2 dataset** [[1]](#1). It contains around 50000 images of indoor spaces along with their grayscale labels, both in resolution 480×640. In order to show results visually, we put
the grayscale pictures through a colormap. Here are some samples from the dataset.

<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188894476-7752fca5-a6ed-43a3-9961-85a5e67ca445.png" width="32%" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188894880-d7a1074c-4636-46a3-8de8-db9e8613d0b6.png" width="32%" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188894956-93fa0a07-2af8-4072-ab4c-8d72622f8918.png" width="32%" />
</p>

<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188895668-fbb4b2e6-3223-4e1f-84fe-a6e9706fdfa2.png" width="32%" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188895721-b878b8b0-2e6b-40eb-a145-b7237c490771.png" width="32%" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188895759-2c90f5fc-9204-4ce2-ac27-d0c5ee00b973.png" width="32%" />
</p>

## U-NET architecture
U-NET was **originally** [[2]](#2) used for image segmentation and it won a Kaggle competition by a large margin. 
As you can see, it has its name because it looks like the letter U. Most of the layers are normal convolution layers, first decreasing dimensions of the image and increasing the number of channels, up until a certain point. From there, upsizing the image until it reaches the starting dimensions while reducing the number of channels. Downsizing is refered to as U-NET Encoder and upsizing U-NET Decoder. The key change that makes U-NET work very well on certain tasks is the connections from encoder layers to same-sized decoder layers. The encoder layer output is simply concatenated on corresponding decoder layer output. This helps bring back the important details from starting layers of U-NET to the decoder part, as downsampling causes loss of sometimes important features.
<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189129087-c6cec4e2-fc87-4493-92c6-ffd512472fc1.png" width="60%" />
</p>

## U-NET first results
We implemented U-NET from scratch, following the dimensions from the original paper. To our surprise, the network itself occupied huge part of CUDA's memory. We managed to train 10 epochs with batch size set to 2, using 40000 images for training. The training was done on NVIDIA Tesla v100 GPU. The results, shown below, were ok, but there were unwanted parallel lines showing. First image is the original one, next to it is the true depth and then our estimation:

<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136462-f6a3aca2-7b7b-4f3e-b08d-7354eaaaec6f.png" width="32%" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136503-8cf13381-bf6a-49ed-b250-7fd238bda8c5.png" width="32%" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136544-148c5395-9b76-40dc-8b08-f67983b5700f.png" width="32%" />
</p>
<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136791-24177112-69d0-4f35-b119-c827a424ad56.png" width="32%" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136817-6ae3b52c-c94b-4a9f-b25a-cd2a80910e41.png" width="32%" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136855-a0200b7b-1147-4a1b-b71e-211162fb0e14.png" width="32%" />
</p>
We figured out the potential reason for the lines - when upsampling the images (decoder part), we used ConvTranspose, but turns out that bilinear Upsampling layer would perform better, the output it produces is more continuous in a way.
However, we found another way to improve our U-NET, still using ConvTranspose.

## U-NET improved version
Inspired by **this paper** [[3]](#3), instead of using only MSE loss, we added together three losses:

* MSE loss 
<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189171469-634379ee-f731-40e0-bb8b-8b96ee87c36b.png" width="300" />
</p>

* Gradient loss
<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189168366-54eb301d-88ff-4c99-aacd-1537dc48f82a.png" width="300" />
</p>
where gradients, or better to say Sobel filters are taken from d - real depth (label) and p - predicted depth

* Normal loss
<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189169194-be9a9d23-31f1-4442-9c98-d22953c20e8a.png" width="300" />
</p>
or equivalently
<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189176779-ff42865e-18ea-4281-9ee9-cd0bce6f1068.png" width="300" />
</p>
which tells us that we want to minimize the angle between gradient vectors of predicted depth and ground truth - when that angle is close to 0, cosines above will be close to one so normal loss will be close to 0. This loss is believed to help with refining details.  
<br />
<br />
We started with our pretrained U-NET on 10 epochs without these losses, and adding them to the game changed the results after only one more epoch! Here are the results, first - original picture, next - true depth, last - our estimation:

<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136462-f6a3aca2-7b7b-4f3e-b08d-7354eaaaec6f.png" width="32%" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136503-8cf13381-bf6a-49ed-b250-7fd238bda8c5.png" width="32%" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189177964-f5f0b93e-3c68-44c9-b83a-b01200e5943a.png" width="32%" />
</p>
<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136791-24177112-69d0-4f35-b119-c827a424ad56.png" width="32%" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136817-6ae3b52c-c94b-4a9f-b25a-cd2a80910e41.png" width="32%" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189177989-db8a4c39-c13a-40b5-8263-41d0afb341df.png" width="32%" />
</p>
As we can see, checkerboard artifacts aren't visible anymore!

## U-NET with pretrained ResNet50 Encoder
To improve our model even more, we took the pretrained ResNet50 from TorchVision model library. **This model** [[4]](#4) from GitHub implemented exactly what we wanted, so we copied their implementation. This model, since using pretrained encoder, didn't take as much space on our CUDA, so we managed to train 10 epochs with batch_size 4, again using 40000 images. Even though the ResNet50 was pretrained on image classification task, this model proved to work really well for depth estimation. The results, again - image, ground truth, our prediction, respectively:
<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136462-f6a3aca2-7b7b-4f3e-b08d-7354eaaaec6f.png" width="32%" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136503-8cf13381-bf6a-49ed-b250-7fd238bda8c5.png" width="32%" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189183664-4280e466-dbbc-4315-a152-787d78d4ee4a.png" width="32%" />
</p>
<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136791-24177112-69d0-4f35-b119-c827a424ad56.png" width="32%" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189136817-6ae3b52c-c94b-4a9f-b25a-cd2a80910e41.png" width="32%" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/189183688-1b07e1d4-c4b4-4e7a-8e5b-b784d6ac26a3.png" width="32%" />
</p>

## Possible improvements
* As mentioned above, bilinear upsampling is expected to do better than convTranspose in the decoder.
* As we could see, choice of loss function is really important - as some papers claim, it would possibly be better to use BerHu loss instead of MSE loss
* The results are much better when using pretrained models - VGG16 encoder with DispNet decoder may produce the best result according to **this paper** [[5]](#5) 

## References
<a id="1">[1]</a>
NYU Depth V2 dataset
https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

<a id="2">[2]</a>U-Net: 
Convolutional Networks for Biomedical Image Segmentation 
https://arxiv.org/pdf/1505.04597.pdf

<a id="3">[3]</a>
Visualization of Convolutional Neural Networks for Monocular Depth Estimation 
https://arxiv.org/pdf/1904.03380v1.pdf

<a id="4">[4]</a>
Implementation of U-NET with ResNet50 encoder 
https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder

<a id="5">[5]</a>
Towards Good Practice for CNN-Based Monocular Depth Estimation
https://openaccess.thecvf.com/content_WACV_2020/papers/Fang_Towards_Good_Practice_for_CNN-Based_Monocular_Depth_Estimation_WACV_2020_paper.pdf
