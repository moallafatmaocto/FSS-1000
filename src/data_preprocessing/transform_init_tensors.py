import cv2
import numpy as np
import torch


def transform_to_tensors(query_images, query_labels, support_images, support_labels, zeros):
    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    support_images_tensor = torch.cat((support_images_tensor, support_labels_tensor), dim=1)
    zeros_tensor = torch.from_numpy(zeros)
    query_images_tensor = torch.from_numpy(query_images)
    query_images_tensor = torch.cat((query_images_tensor, zeros_tensor), dim=1)
    query_labels_tensor = torch.from_numpy(query_labels)
    return query_images_tensor, query_labels_tensor, support_images_tensor, support_labels_tensor


def init_arrays(class_num, sample_num_per_class, batch_num_per_class):
    support_images = np.zeros((class_num * sample_num_per_class, 3, 224, 224), dtype=np.float32)
    support_labels = np.zeros((class_num * sample_num_per_class, class_num, 224, 224), dtype=np.float32)
    query_images = np.zeros((class_num * batch_num_per_class, 3, 224, 224), dtype=np.float32)
    query_labels = np.zeros((class_num * batch_num_per_class, class_num, 224, 224), dtype=np.float32)
    zeros = np.zeros((class_num * batch_num_per_class, 1, 224, 224), dtype=np.float32)
    class_cnt = 0
    return class_cnt, query_images, query_labels, support_images, support_labels, zeros


def normalize_and_transpose(image):
    image = image[:, :, ::-1]  # bgr to rgb
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    return image


def import_image_and_label(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        print(image_path)
        raise Exception('cannot load image ')

    if np.shape(image)[1] != 224 or np.shape(image)[0] != 224:
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    image = normalize_and_transpose(image)
    label = cv2.imread(label_path)[:, :, 0]
    if np.shape(label)[1] != 224 or np.shape(label)[0] != 224:
        label = cv2.resize(label, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    return image, label
