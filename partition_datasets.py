import csv
import random
from typing import Dict, List
from pathlib import Path
import argparse


def create_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # path to directory with folders: bin, csv, results, and models 
    parser.add_argument("-proj_dir", type=Path)
    # proportion of samples of the total dataset to put into validation and test partitions
    parser.add_argument("-val_test_prop", type=float, default=0.1)
    # number of classes the model outputs
    parser.add_argument("-num_classes", type=int, default=2)
    return parser.parse_args()


def get_class_dict(proj_dir: Path) -> Dict:
    class_dict = {}
    class_dict[0] = [x.parts[-1] for x in proj_dir.joinpath("no").iterdir()]
    class_dict[1] = [x.parts[-1] for x in proj_dir.joinpath("yes").iterdir()]
    return class_dict


def get_sample(class_dict: Dict, amount: int, class_idx: int) -> List:
    return random.sample(class_dict[class_idx], amount)


def add_to_dict(part_dict: Dict, class_idx: int, sample_list: List) -> Dict:
    part_dict[class_idx].extend(sample_list)
    return part_dict


def delete_sample(class_dict: Dict, sample_list: List, class_idx: int) -> Dict:
    class_dict[class_idx] = [x for x in class_dict[class_idx] if x not in sample_list]
    return class_dict
 

def partition_datasets(
    val_test_prop: float, 
    class_dict: Dict, 
    num_classes: int
    ) -> tuple([Dict, Dict, Dict]):
    val_dict = {0: [], 1: []}
    test_dict = {0: [], 1: []}

    for class_idx in range(num_classes):
        val_sample = get_sample(
            class_dict=class_dict,
            amount=int(val_test_prop*len(class_dict[class_idx])),
            class_idx=class_idx
            )
        val_dict = add_to_dict(part_dict=val_dict, class_idx=class_idx, sample_list=val_sample)
        class_dict = delete_sample(class_dict=class_dict, sample_list=val_sample, class_idx=class_idx)
        test_sample = get_sample(
            class_dict=class_dict,
            amount=int(val_test_prop*len(class_dict[class_idx])),
            class_idx=class_idx
            )
        test_dict = add_to_dict(part_dict=test_dict, class_idx=class_idx, sample_list=test_sample)
        class_dict = delete_sample(class_dict=class_dict, sample_list=test_sample, class_idx=class_idx)

    return class_dict, val_dict, test_dict


def write_csv(part_dict: Dict, filepath: Path) -> None:
    with open(filepath, mode="w", newline="") as opened_csv:
        writer = csv.writer(opened_csv)
        for file_class, file_name_list in part_dict.items():
            for file_name in file_name_list:
                writer.writerow((file_name, file_class))


def main():
    args = create_argparser()
    class_dict = get_class_dict(proj_dir=args.proj_dir)
    train_dict, val_dict, test_dict = partition_datasets(
        val_test_prop=args.val_test_prop, class_dict=class_dict, num_classes=args.num_classes
        )
    write_csv(part_dict=train_dict, filepath=args.proj_dir.joinpath("train.csv"))
    write_csv(part_dict=val_dict, filepath=args.proj_dir.joinpath("val.csv"))
    write_csv(part_dict=test_dict, filepath=args.proj_dir.joinpath("test.csv"))


if __name__ == "__main__":
    main()
