# basic-image-classification

This is an example of image classification with PyTorch. 
[Disclaimer]: It should be noted that this model was trained / evaluated on 252 images and even though it shows impressive performance, it can't be assumed the model will work the same in a clinical setting without developing a larger dataset with as little bias introduced as possible. This is also meant to be as simple of an example as possible and doesn't include channel-wise pixel centering and normalization, transfer learning, fine tuning, or modern architectures such as EfficientNet.

A small dataset from Kaggle was used to train convolutional neural networks to classify brain MRI images as having a malignancy or not. 
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

| Number of training images  | Number of validation images | Number of test images |
| -------------------------- | --------------------------- | --------------------- |
| 202                        | 25                          | 26                    |

| Positive training images  | Negative training images | Positive validation images | Negative validation images | Positive test images | Negative test images |
| ------------------------- | ------------------------ | -------------------------- | -------------------------- | -------------------- | -------------------- |
| 124                       | 78                       | 15                         | 10                         | 16                   | 10                   |

![1 no](https://user-images.githubusercontent.com/118086192/215293784-508ed065-5553-4983-a4b2-bc7fe4d867bc.jpeg)

Example negative image (above).

![Y33](https://user-images.githubusercontent.com/118086192/215293814-cfaf96e0-ead9-4a5d-b550-2102e56fddd3.jpg)

Example positive image (above).

The images were re-sized to (100, 100) and were randomly partitioned into the training, validation and test sets.

Three different architectures were used: ResNet-18, VGG-11, and DenseNet-121 from torchvision.

![3d_hyperparameter_plot](https://user-images.githubusercontent.com/118086192/215295101-2c725566-c25a-4beb-b241-5e145c0f7691.PNG)

Plot of the batch size and learning rate hyperparameters and minimum validation loss achieved during training. 
