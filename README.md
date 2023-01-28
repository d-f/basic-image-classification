# basic-image-classification

This is an example of image classification with PyTorch. 
[Disclaimer]: It should be noted that this model was trained / evaluated on 252 images and even though it shows impressive performance, it can't be assumed the model will work the same in a clinical setting without developing a large dataset with as little bias introduced as possible. This is also meant to be as simple of an example as possible, and doesn't include channel-wise pixel centering and normalization, transfer learning, fine tuning, or modern architectures such as EfficientNet.

A small dataset from Kaggle was used to train a model to classify MRI images as having a malignancy or not. 
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?resource=download

| Number of training images  | Number of validation images | Number of test images |
| -------------------------- | --------------------------- | --------------------- |
| 202                        | 25                          | 26                    |
