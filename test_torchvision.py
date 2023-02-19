from sklearn.preprocessing import OneHotEncoder
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.models as models
import torch.nn as nn


def create_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # project directory with folders: bin, csv, results, and models
    parser.add_argument("-dir", "--project_directory", type=Path)
    # number of classes the model outputs  
    parser.add_argument("-classes", "--num_classes", type=int) 
    # number of images to take as input at once
    parser.add_argument("-batch_size", "--batch_size", type=int) 
    # name of the model being trained e.g. model.pth.tar
    parser.add_argument("-save", "--model_load_name", type=str)
    # one of {"resnet18", "vgg", "densenet"}
    parser.add_argument("-architecture", type=str) 
    # name of the JSON file to save the results to
    parser.add_argument("-result_json_name", type=str)
    # if set to true, this argument uses the GPU
    # to set as true use --use_GPU in CLI 
    parser.add_argument("-use_GPU", action="store_true")
    # size that the images will be resized to
    parser.add_argument("-img_size", nargs="+")
    
    return parser.parse_args()


def define_device(use_GPU: bool) -> torch.device:
    """
    defines device to manage allocation of tensors
    """
    if use_GPU:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def define_model(
    num_classes: int,
    architecture: str,
    device: torch.device
    ) -> models:
    if architecture == "resnet18":
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
        return model.to(device)
    elif architecture == "vgg":
        model = models.vgg11_bn()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        return model.to(device)
    elif architecture == "densenet":
        model = models.densenet121()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        return model.to(device)


def load_model(weight_path: Path, model: models) -> models:
    '''
    loads all parameters of a model
    :param weight_path: path to the .pth.tar file with parameters to update
    :param model: model object
    :return: the model with updated parameters
    '''
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model


class ImageClassificationDataset(Dataset):
    '''
    dataset reads filenames and classes
    from csv and returns an opened image, the class and the filename
    '''

    def __init__(self, csv_file, root_dir, transform=None, resize=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))  # 2nd column is label integer
        img_path = self.root_dir.joinpath(
            self.annotations.iloc[index, 0])  # row i column 0 since its the column with name in csv file
        image = Image.open(img_path).convert("RGB")
        # convert [["2", "2", "4"], ["2", "2", "4"]]
        # to (224, 224)
        resize_formatted_tuple = (strings_to_int(x) for x in self.resize)
        if self.resize:
            image = image.resize(size=resize_formatted_tuple)
        if self.transform:
            image = self.transform(image)

        return image, y_label


def strings_to_int(string_list: list) -> int:
    empty_string = ""
    return int(empty_string.join(string_list))


def create_datasets(batch_size: int, test_file_path: Path, root_dir: Path, img_size: tuple) -> torch.utils.data.DataLoader:
    '''
    reads dataset mean and standard deviation for each channel
    creates the datasets and returns dataloaders
    '''

    test_dataset = ImageClassificationDataset(
        csv_file=test_file_path,
        root_dir=root_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        resize=img_size[1:]
    )
    return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def save_preds(
    labels: list, 
    preds: list, 
    pred_prob_list: list, 
    label_prob_list: list, 
    filepath: Path
    ) -> None:
    with open(filepath, mode="w") as opened_json:
        json_obj = {
            "labels": labels,
            "predictions": preds,
            "predictive probabilities": pred_prob_list,
            "label probabilities": label_prob_list
        }
        json.dump(json_obj, opened_json)


def test_best_model(
    model_load_name: str, 
    num_classes: int, 
    loader: torch.utils.data.DataLoader, 
    model: models, 
    model_save_dir: Path, 
    device: torch.device
    ) -> tuple([list, list, list, list]):
    """
    evaluates best performing model on test set
    as dictated by best_checkpoint based on individual
    image tiles and performance based on correctly predicted
    slides when all predictions within a slide are averaged
    """
    with torch.no_grad():
        load_model(weight_path=model_save_dir.joinpath(model_load_name), model=model)
        label_list = []
        prediction_list = []
        pred_prob_list = []
        label_prob_list = []
        model.eval()
        onehot_encoder = OneHotEncoder(sparse=False)
        # fit encoder to a list of all classes from 0 to num_classes - 1
        # reshaped from [0, 1, 2, 3] to [[0], [1], [2], [3]]
        onehot_encoder = onehot_encoder.fit(
            np.array([x for x in range(num_classes)]).reshape(-1, 1)
        )

        for imgs, labels in tqdm(loader, desc="Test"):
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            scores = model(imgs)
            label_list += [int(x) for x in labels]
            prediction_list += [int(np.argmax(x)) for x in scores]
            pred_prob_list += [tuple(x.cpu().numpy().astype(np.float64)) for x in scores]
            # labels is a list of tensors, so they need to be placed onto the cpu, converted to numpy arrays, 
            # reshaped with .reshape(-1, 1) to go from [1, 0] to [[1, 0]], transformed with onehot encoder, 
            # and indexed with [0] to go from [[0, 1]] to [0, 1]. then converted to a tuple to be JSON serialized
            label_prob_list += [tuple(onehot_encoder.transform(x.cpu().numpy().reshape(-1, 1))[0]) for x in labels]

        return label_list, prediction_list, label_prob_list, pred_prob_list


def main():
    args = create_argparser()
    test_loader = create_datasets(
        batch_size=args.batch_size,
        test_file_path=args.project_directory.joinpath('csv').joinpath("test.csv"), 
        root_dir=args.project_directory.joinpath('bin'),
        img_size=args.img_size
        )
    device = define_device(use_GPU=args.use_GPU)
    model = define_model(num_classes=args.num_classes, architecture=args.architecture, device=device)
    labels, preds, label_probs, pred_probs = test_best_model(
        model_load_name=args.model_load_name,
        num_classes=args.num_classes,
        loader=test_loader, 
        model=model, 
        model_save_dir=args.project_directory.joinpath("models"), 
        device=device
        )
    save_preds(
        labels=labels, 
        preds=preds, 
        filepath=args.project_directory.joinpath(args.result_json_name), 
        pred_prob_list=pred_probs, 
        label_prob_list=label_probs
        )


if __name__ == "__main__":
    main()
