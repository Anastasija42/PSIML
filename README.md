# PSIML - DEPTH ESTIMATION PROJECT
The goal of our project is monocular depth estimation using U-NET stucture.
First, we implemented U-NET from scratch, but the results weren't the best.
To improve them, we changed the loss, from usual MSE loss to sum of three losses: MSE, gradient and normal loss.
For even better results, we substituted U-NET encoder with pretrained ResNet50.
## Dataset
All our models were trained on NYU Depth V2 dataset. It contains around 50000 images of indoor spaces along with their grayscale labels. In order to show results visually, we put
the grayscale pictures through colormap. Here are some samples from the dataset.

<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188894476-7752fca5-a6ed-43a3-9961-85a5e67ca445.png" width="300" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188894880-d7a1074c-4636-46a3-8de8-db9e8613d0b6.png" width="300" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188894956-93fa0a07-2af8-4072-ab4c-8d72622f8918.png" width="300" />
</p>

<p float="left" align="middle">
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188895668-fbb4b2e6-3223-4e1f-84fe-a6e9706fdfa2.png" width="300" />
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188895721-b878b8b0-2e6b-40eb-a145-b7237c490771.png" width="300" /> 
  <img align="top" src="https://user-images.githubusercontent.com/112171137/188895759-2c90f5fc-9204-4ce2-ac27-d0c5ee00b973.png" width="300" />
</p>




