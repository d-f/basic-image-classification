# basic-image-classification

This is an example of image classification with PyTorch. 
[Disclaimer]: It should be noted that this model was trained / evaluated on 253 images and even though it shows impressive performance, it can't be assumed the model will work the same in a clinical setting without developing a larger dataset with as little bias introduced as possible. This is also meant to be as simple of an example as possible and doesn't include channel-wise pixel centering and normalization, transfer learning, fine tuning, inspecting model predictions via Grad-CAM / saliency maps / visualizing attention, measuring model uncertainty via Monte Carlo simulations or using modern architectures such as EfficientNet or Vision Transformers.

A small dataset from Kaggle was used to train convolutional neural networks to classify brain MRI images as having a malignancy or not. 
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

| Number of training images  | Number of validation images | Number of test images |
| -------------------------- | --------------------------- | --------------------- |
| 202                        | 25                          | 26                    |

Table 1: Total number of images in the different dataset partitions.

| Positive training images  | Negative training images | Positive validation images | Negative validation images | Positive test images | Negative test images |
| ------------------------- | ------------------------ | -------------------------- | -------------------------- | -------------------- | -------------------- |
| 124                       | 78                       | 15                         | 10                         | 16                   | 10                   |

Table 2: Number of images in each class for the different dataset partitions.

![1 no](https://user-images.githubusercontent.com/118086192/215293784-508ed065-5553-4983-a4b2-bc7fe4d867bc.jpeg)

Figure 1: Example negative image (above).

![Y33](https://user-images.githubusercontent.com/118086192/215293814-cfaf96e0-ead9-4a5d-b550-2102e56fddd3.jpg)

Figure 2: Example positive image (above).

The images were re-sized to (224, 224) and were randomly partitioned into the training, validation and test sets.

Three different architectures were used: ResNet-18, VGG-11, and DenseNet-121 from torchvision.

![3d_hyperparameter_plot](https://user-images.githubusercontent.com/118086192/215295101-2c725566-c25a-4beb-b241-5e145c0f7691.PNG)

Figure 3: Plot of the batch size, learning rate and minimum validation loss achieved during training for ResNet-18 (Blue), VGG-11 (Green) and DenseNet-121 (Red). ResNet-18 achieved the lowest loss value, but on average DenseNet-121 achieved lower than the averages of other architectures. 

| Batch size  | Learning rate | Number of epochs | Optimizer | Loss          | Patience |
| ----------- | ------------- | ---------------- | --------- | ------------- | -------- | 
| 25          | 0.001         |  13              | Adam      | Cross Entropy | 5        |

Table 3: Batch size, learning rate, total number of training epochs, optimizer algorithm, loss function and patience for the best performing model (ResNet-18). Batch size refers to the number of inputs processed before the parameters are updated, learning rate is the proportion of the gradient that is used to update the parameters, number of training epochs is the total number of times the training goes through the entire dataset, optimizer is the algorithm used to calculate the parameter updates, loss function is the funciton used to measure the error between prediction and ground truth, and patience is the total number of epochs the training went past the model achieving a minimum validation loss value. Parameters are saved at the minimum validation loss and the model evaluated on the test set. 

![resnet_model_1_accuracy_curve](https://user-images.githubusercontent.com/118086192/215297118-e7932cd4-8cc6-4066-b6b9-006799168412.png)

Figure 4: Accuracy on the train and validation datasets throughout training, including the extra 5 epochs the model was trained past the minimum validation loss.

![resnet_model_1_loss_curve](https://user-images.githubusercontent.com/118086192/215297135-acf8d105-ce25-4c06-a589-9cd1f1e6f15f.png)

Figure 5: Loss on the train and validation datasets throughout training, including the extra 5 epochs the model was trained past the minimum validation loss.

| Sensitivity (Recall) | Specificity | ROC-AUC | Accuracy | 
| -------------------- | ----------- | ------- | -------- |  
| 100%                 | 80%         |  0.9    | 92%      | 

Table 4: Model performance on the test dataset.

