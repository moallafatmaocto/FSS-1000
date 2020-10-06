import os
import random
import shutil
from typing import Set, Tuple, List

from src.data_preprocessing.config import DATA_PATH, TRAIN_NUMBER, TEST_CATEGORIES


def get_random_n_classes(class_list: List, n: int) -> List:
    # For a randomly selected train/test
    classes = list(range(0, len(class_list)))
    print('total number of initial classes', len(classes))
    chosen_classes = random.sample(classes, n)
    return chosen_classes


def get_train_and_val_image_list_randomly(class_list: List, train_class_number: int) -> Tuple[List, List]:
    # For a randomly selected train/test
    print('class_list', class_list)
    chosen_train_classes = [class_list[idx] for idx in (get_random_n_classes(class_list, train_class_number))]
    print('Number of train classes', len(chosen_train_classes))
    chosen_val_classes = [class_name for class_name in class_list if (class_name not in (chosen_train_classes))]
    print('Number of test classes', len(chosen_val_classes), chosen_val_classes)
    return chosen_train_classes, chosen_val_classes


def get_train_val_test_split_from_file(data_directory_path: str, test_categories: Set[str]) -> Tuple[
    List[str], List[str], List[str]]:
    all_categories: Set[str] = set([category for category in os.listdir(data_directory_path)
                                    if category != '.DS_Store'])
    assert len(all_categories) == 1000

    train_val_categories = list(all_categories.difference(test_categories))
    chosen_train_classes, chosen_val_classes = get_train_and_val_image_list_randomly(train_val_categories,
                                                                                     train_class_number=TRAIN_NUMBER)
    return chosen_train_classes, chosen_val_classes, list(test_categories)


def create_images_directories(base_path: str, directory: str) -> str:
    path = os.path.join(base_path, directory)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def move_to_new_dir(classes, original_path, new_path):
    print('Start Moving ...')
    for class_name in classes:
        print('Moving class ', class_name)
        print('current', os.getcwd(), 'original', original_path)
        shutil.move(os.path.join(original_path, class_name), new_path)
    print('Moving ended.')


if __name__ == '__main__':
    base_path = DATA_PATH
    test_categories = TEST_CATEGORIES
    train_classes, val_classes, test_classes = get_train_val_test_split_from_file(base_path, test_categories)
    train_path = create_images_directories(base_path, 'train')
    val_path = create_images_directories(base_path, 'val')
    test_path = create_images_directories(base_path, 'test')
    move_to_new_dir(train_classes, base_path, train_path)
    move_to_new_dir(val_classes, base_path, val_path)
    move_to_new_dir(test_classes, base_path, test_path)
