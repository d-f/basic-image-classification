import csv
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import argparse
from pathlib import Path
from PIL import Image


def create_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # directory with relevant folders
    parser.add_argument("-project_directory", type=Path)
    # number of epochs before training is stopped if patience hasn't stopped it early  
    parser.add_argument("-num_epochs", type=int)
    # number of classes within the dataset
    parser.add_argument("-num_classes", type=int)
    # proportion of the gradient that is used to update parameters
    parser.add_argument("-learning_rate", type=float)
    # training will stop when this number of epochs have past 
    # and the validation loss has not improved
    parser.add_argument("-patience", type=int)  
    # number of images to take feed into the model before 
    # measuring gradient and updating parameters
    parser.add_argument("-batch_size", type=int)
    # name of the model being trained e.g. model.pth.tar
    parser.add_argument("-model_save_name", type=str)  
    # shape of the input image -> channels first for PyTorch
    parser.add_argument("-img_shape", type=list, nargs="+") 
    # architecture to use
    parser.add_argument("-architecture", type=str)
    # if set to true, this argument uses the GPU
    # to set as true use --use_GPU in CLI
    parser.add_argument("-use_GPU", action="store_true")
    return parser.parse_args()


def read_csv(csv_path: Path) -> List:
    with open(csv_path) as opened_csv:
        reader = csv.reader(opened_csv)
        return [x for x in reader]


def define_device(use_GPU: bool) -> torch.device:
    """
    defines device to manage allocation of tensors
    """
    if use_GPU:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def define_model(
    architecture: str,
    num_classes: int,
    device: torch.device
    ) -> models:
    """
    defines different torchvision models and replaces the classifier layer with an
    appropriatiely sized one for the number of classes in the dataset
    """
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
        


def define_optimizer(model: models.efficientnet.EfficientNet, learning_rate: float) -> optim.Adam:
    """
    returns the algorithm that updates the parameters
    """
    return optim.Adam(model.parameters(), lr=learning_rate)


def define_criterion() -> nn.CrossEntropyLoss:
    """
    returns the algorithm to measure loss 
    """
    return nn.CrossEntropyLoss()


def load_model(weight_path: Path, model: models.efficientnet.EfficientNet) -> models.efficientnet:
    """
    loads all parameters of a model
    """
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model


class ImageClassificationDataset(Dataset):
    """
    dataset reads filenames and classes
    from csv and returns an opened image and the class
    """
    def __init__(self, csv_file: Path, img_dir: Path, transform=None, resize=None) -> None:
        self.dataset_tuples = read_csv(csv_path=csv_file) # (file name, class)
        self.img_dir = img_dir
        self.transform = transform
        self.resize = resize

    def __len__(self) -> None:
        return len(self.dataset_tuples)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        y_label = torch.tensor(int(self.dataset_tuples[index][1]))  
        img_path = self.img_dir.joinpath(self.dataset_tuples[index][0])  
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


def save_checkpoint(state: Dict, filepath: Path) -> None:
    """
    saves the model state dictionary to a .pth.tar tile
    """
    print("saving...")
    torch.save(state, filepath)


def create_datasets(
    img_size: tuple,
    batch_size: int,
    train_file_path: Path, 
    img_dir: Path, 
    val_file_path: Path
    ) -> tuple[
        ImageClassificationDataset, 
        torch.utils.data.DataLoader,
        ImageClassificationDataset, 
        torch.utils.data.DataLoader, 
        List, 
        List
    ]:
    """
    creates the datasets and returns dataloaders
    """

    train_dataset = ImageClassificationDataset(
        csv_file=train_file_path,
        img_dir=img_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        resize=img_size[1:]
    )
    val_dataset = ImageClassificationDataset(
        csv_file=val_file_path,
        img_dir=img_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        resize=img_size[1:]
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, train_loader, val_dataset, val_loader


@torch.no_grad()
def validate_model(
    model: models.efficientnet.EfficientNet, 
    val_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    criterion: nn.CrossEntropyLoss, 
    val_losses: List, 
    num_correct_val: int
    ) -> tuple[models.efficientnet_b0, List, int]:
    """
    evaluates model performance on the validation set
    """
    model.eval()  # turn dropout and batch norm off
    for val_data, val_targets in tqdm(val_loader, desc='Validation'):
        val_data = val_data.to(device=device)
        val_targets = val_targets.to(device=device)
        val_scores = model(val_data)
        _, val_predictions = val_scores.max(1)
        num_correct_val += (val_predictions == val_targets).sum().detach()
        val_loss = criterion(val_scores, val_targets)
        val_losses.append(val_loss.item())
    model.train()  # turn regularization back on
    return model, val_losses, num_correct_val


def train(
    num_epochs: int,
    patience: int, 
    model_save_name: str,
    train_dataset: ImageClassificationDataset, 
    train_loader: torch.utils.data.DataLoader, 
    val_dataset: ImageClassificationDataset, 
    val_loader: torch.utils.data.DataLoader, 
    model: models.efficientnet, 
    optimizer: optim.Adam, 
    device: torch.device, 
    criterion: nn.CrossEntropyLoss, 
    model_save_dir: Path
    ) -> tuple[List, List, List, List]:
    """
    trains a model until the validation
    loss fails to decrease after a specified number of epochs [patience]
    """
    patience_counter = 0
    best_checkpoint = None
    plot_losses = []
    plot_val_losses = []
    min_val_loss = 0
    plot_accuracy = []
    plot_val_accuracy = []

    for epoch in range(num_epochs):
        if patience_counter == patience:
            break  # early stopping
        losses = []
        val_losses = []
        num_correct_train = 0
        num_correct_val = 0
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

        for data, targets in tqdm(train_loader, desc='Train'):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)

            with torch.no_grad():  # counting correct predictions
                _, predictions = scores.max(1)
                num_correct_train += (predictions == targets).sum().detach()

            loss = criterion(scores, targets)  # calculate loss
            losses.append(loss.item())
            optimizer.zero_grad()  # clear gradient information
            loss.backward()  # calculate gradient
            optimizer.step()

        model, val_losses, num_correct_val = validate_model(
            model=model, 
            val_loader=val_loader, 
            device=device, 
            criterion=criterion, 
            val_losses=val_losses, 
            num_correct_val=num_correct_val
            )

        if epoch == 0:
            min_val_loss = (sum(val_losses) / len(val_losses))
            best_checkpoint = checkpoint
            save_checkpoint(state=best_checkpoint, filepath=model_save_dir.joinpath(model_save_name))
            patience_counter += 1
        else:
            new_loss = (sum(val_losses) / len(val_losses))
            if new_loss < min_val_loss:
                best_checkpoint = checkpoint
                save_checkpoint(state=best_checkpoint, filepath=model_save_dir.joinpath(model_save_name))
                min_val_loss = (sum(val_losses) / len(val_losses))
                patience_counter = 0
            else:
                patience_counter += 1

        # print loss, accuracy, val loss, and val accuracy after end of epoch
        print(f'Epoch: {epoch + 1} Loss: {sum(losses) / len(losses)}\
        Accuracy: {num_correct_train.cpu().numpy() / len(train_dataset)}\
        Validation Loss: {sum(val_losses) / len(val_losses)}\
        Validation Accuracy: {num_correct_val.cpu().numpy() / len(val_dataset)}', flush=True)

        plot_losses.append([epoch + 1, (sum(losses) / len(losses))])
        plot_val_losses.append([epoch + 1, (sum(val_losses) / len(val_losses))])
        plot_accuracy.append([epoch + 1, (num_correct_train.cpu().numpy() / len(train_dataset))])
        plot_val_accuracy.append([epoch + 1, (num_correct_val.cpu().numpy() / len(val_dataset))])

    return plot_losses, plot_val_losses, plot_accuracy, plot_val_accuracy


def save_results(
        batch_size: int,
        model_save_name: str,
        train_tile_csv: str,
        val_tile_csv: str,
        patience: int,
        plot_losses: List, 
        plot_val_losses: List, 
        plot_accuracy: List,
        plot_val_accuracy: List, 
        result_dir: Path, 
        optimizer: optim.Adam
) -> None:
    """
    saves the results of training and hyperparameters into json files
    """
    # save hyperparameter configurations
    hyperparameters = {'batch size': batch_size,
                       'model save name': model_save_name,
                       'optimizer': optimizer.defaults, 
                       'train dataset': train_tile_csv,
                       'validation dataset': val_tile_csv,
                       'patience': patience}

    json_losses = {'loss values': plot_losses,
                   'accuracy values': plot_accuracy,
                   'val loss values': plot_val_losses,
                   'val accuracy values': plot_val_accuracy}

    with open(f'{result_dir.joinpath(model_save_name[:-8])}_hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile)

    with open(f'{result_dir.joinpath(model_save_name[:-8])}_loss_values.json', 'w') as outfile:
        json.dump(json_losses, outfile)


def main():
    args = create_argparser()
    train_dataset, train_loader, val_dataset, val_loader = create_datasets(
        batch_size=args.batch_size,
        img_size=args.img_shape,
        train_file_path=args.project_directory.joinpath('csv').joinpath("train.csv"), 
        img_dir=args.project_directory.joinpath('bin'), 
        val_file_path=args.project_directory.joinpath("csv").joinpath("val.csv")
        )

    device = define_device(use_GPU=args.use_GPU)
    model = define_model(architecture=args.architecture, num_classes=args.num_classes, device=device)
    optimizer = define_optimizer(model=model, learning_rate=args.learning_rate)
    criterion = define_criterion()

    plot_losses, plot_val_losses, plot_accuracy, plot_val_accuracy = train(
        num_epochs=args.num_epochs, 
        patience=args.patience, 
        model_save_name=args.model_save_name, 
        train_dataset=train_dataset, 
        train_loader=train_loader, 
        val_dataset=val_dataset, 
        val_loader=val_loader, 
        model=model, 
        optimizer=optimizer, 
        device=device, 
        criterion=criterion, 
        model_save_dir=args.project_directory.joinpath("models")
    )
    save_results(
        batch_size=args.batch_size, 
        model_save_name=args.model_save_name,
        train_tile_csv="train.csv",
        val_tile_csv="val.csv",
        patience=args.patience,
        plot_losses=plot_losses, 
        plot_val_losses=plot_val_losses, 
        plot_accuracy=plot_accuracy,
        plot_val_accuracy=plot_val_accuracy, 
        result_dir=args.project_directory.joinpath("results"), 
        optimizer=optimizer
    )


if __name__ == "__main__":
    main()
    

    
