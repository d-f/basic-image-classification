# basic-image-classification

This is an example of image classification with PyTorch. 

In order to run and re-create the results presented:
- **Set up environment:**
  - Download and extract the kaggle dataset [https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) to a directory, in this example C:\mlprojects\kaggle_brain_classification\
  - Clone the basic-image-classification repository to a directory, in this case C:\mlprojects\
  - Create conda environment, in this case named basic_image_env
  ```
  cd C:\mlprojects\
  git clone https://github.com/d-f/basic-image-classification.git
  conda create -n basic_image_env python=3
  conda activate basic_image_env
  pip install -r C:\mlprojects\basic_image_classification\requirements.txt
  ```
- **Create dataset CSV files**
  - Run "partition_datasets.py" to create CSV files for PyTorch dataset/ dataloader creation. This needs to be done before the bash / powershell scripts are run since they rely on determining image class via folder location (e.g. "yes" vs "no").
  ```
  python C:\mlprojects\basic_image_classification\partition_datasets.py -proj_dir C:\mlprojects\kaggle_brain_classification -val_test_prop 0.1 -num_classes 2
  ```
- **Organize images**
  - Depending on the OS, run organize_files.ps1 or organize_files.sh to organize images, create folders and organize csv files
  ```
  C:\mlprojects\basic_image_classification\organize_files.ps1 "C:\mlprojects\kaggle_brain_classification\"
  ```
  or 
  ```
  /mlprojects/basic_image_classification/organize_files.sh
  "/mlprojects/basic_image_classification/"
  ```
- **Train model**
  - Run "train_torchvision.py" --use_GPU sets use_GPU as true, ommiting this argument sets use_GPU as False
  ```
    python C:\mlprojects\basic_image_classification\train_torchvision.py -project_directory C:\mlprojects\kaggle_brain_classification\ -num_epochs 256 -num_classes 2 -learning_rate 0.001 -patience 5 -batch_size 25 -model_save_name resnet_1.pth.tar -img_shape 3 224 224 -architecture resnet18 --use_GPU
  ```
- **Test model**
  - Run "test_torchvision.py"
  ```
    python C:\mlprojects\basic_image_classification\test_torchvision.py -dir C:\mlprojects\kaggle_brain_classification\ -classes 2 -batch_size 100 -save resnet_1.pth.tar -architecture resnet18 -result_json_name resnet_1_preds.json -img_size 3 224 224 --use_GPU
  ```
- **Calculate model performance**
  - Run "calc_model_performance.R", in this case for a model prediction file named resnet_1_preds.json and a performance json to be named "resnet_1_results.json"
  ```
    Rscript C:\mlprojects\basic_image_classification\calc_model_performance.R
    C:\mlprojects\kaggle_brain_classification\resnet_1_preds.json
    C:\mlprojects\kaggle_brain_classification\resnet_1_results.json
  ```
***Results of the best performing model presented below:***

***[Disclaimer]***: It should be noted that this model was trained / evaluated on 253 images and even though it shows impressive performance, it can't be assumed the model will work the same in a clinical setting without developing a larger dataset with as little bias introduced as possible. This is also meant to be as simple of an example as possible and doesn't include data augmentation, channel-wise pixel centering and normalization, transfer learning, fine tuning, inspecting model predictions via [Grad-CAM](https://arxiv.org/abs/1610.02391) or [visualizing attention](https://arxiv.org/abs/2010.11929), measuring model uncertainty via  [Monte Carlo simulations](https://arxiv.org/pdf/1506.02142.pdf) or using modern architectures such as [EfficientNet](https://arxiv.org/abs/1905.11946) or [Vision Transformers](https://arxiv.org/abs/2010.11929).

A small dataset from Kaggle was used to train convolutional neural networks to classify brain MRI images as containing a malignancy or not. 
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
Since the dataset contains a mixture of single and multi-channel images, all are converted to grayscale multi-channel images.

Three different architectures were used: [ResNet-18](https://arxiv.org/abs/1512.03385), [VGG-11](https://arxiv.org/abs/1409.1556), and [DenseNet-121](https://arxiv.org/abs/1608.06993) from [torchvision](https://pytorch.org/vision/0.8/models.html).

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

| Sensitivity (Recall) | Specificity | ROC-AUC | Accuracy | Cohen's Kappa |
| -------------------- | ----------- | ------- | -------- | ------------- |
| 90.00 %              | 93.75 %     |  1.0    | 92.31 %  | 0.8375        |

Table 4: Model performance on the test dataset.

**Convolutional neural network background:**

Convolutional neural networks work by applying successive convolutional layers to an image to distill it down to a set of feature maps, representing where useful imaging features (such as patterns, shapes, etc.) exist within the image. The filter is applied to the image and moved across with a set value known as stride, which affects the size of the resulting feature map. Convolutional neural networks use the stride to distill down a large image to a set of smaller feature maps. Some known filters can produce effects like highlighting certain imaging features, such as the sobel filter which highlights edges in images. 

Input (source: https://en.wikipedia.org/wiki/Sobel_operator): 

![image](https://github.com/d-f/basic-image-classification/assets/118086192/a892de6e-caa3-4fad-a6f7-4a1dfa0a5e7d)

Output (source: https://en.wikipedia.org/wiki/Sobel_operator):

![image](https://github.com/d-f/basic-image-classification/assets/118086192/b4dd3d4c-2cdd-43f2-b3ae-9aaa820d4d94)


The CNN is able to learn which filters would be the best to highlight the best imaging features for classifying the input image. From the resulting feature maps, a simple neural network known as a fully connected network directly predicts a probability distribution representing the probability that the image is of a certain class. 
