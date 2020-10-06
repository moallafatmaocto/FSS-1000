import os
import random
import numpy as np
import cv2

from src.data_preprocessing.transform_init_tensors import init_arrays, normalize_and_transpose, import_image_and_label, \
    transform_to_tensors


def get_training_batch(class_num, sample_num_per_class, batch_num_per_class,
                       train_data_path):  # shuffle in query_images not done
    # classes.remove(EXCLUDE_CLASS)

    classes_name = os.listdir(train_data_path)
    classes = list(range(0, len(classes_name)))

    chosen_classes = random.sample(classes, class_num)
    class_cnt, query_images, query_labels, support_images, support_labels, zeros = init_arrays(class_num,
                                                                                               sample_num_per_class,
                                                                                               batch_num_per_class)
    for i in chosen_classes:
        # print ('class %s is chosen' % i)
        imgnames = [file.split('.jpg')[0]
                    for file in os.listdir(f'{train_data_path}/{classes_name[i]}')
                    if file.endswith('.jpg')]

        indexs = list(range(0, len(imgnames)))
        chosen_index = random.sample(indexs, sample_num_per_class + batch_num_per_class)
        j = 0
        for k in chosen_index:
            # process image
            image_path = f"{train_data_path}/{classes_name[i]}/{imgnames[k]}.jpg"
            label_path = f"{train_data_path}/{classes_name[i]}/{imgnames[k]}.png"
            image, label = import_image_and_label(image_path, label_path)

            if j < sample_num_per_class:
                support_images[j] = image
                support_labels[j][0] = label
            else:
                query_images[j - sample_num_per_class] = image
                query_labels[j - sample_num_per_class][class_cnt] = label
            j += 1

        class_cnt += 1
    query_images_tensor, query_labels_tensor, support_images_tensor, support_labels_tensor = transform_to_tensors(
        query_images, query_labels, support_images, support_labels, zeros)

    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor, chosen_classes


def get_autolabel_batch(testname, class_num, sample_num_per_class,
                        batch_num_per_class, support_dir, test_dir):
    class_cnt, query_images, query_labels, support_images, support_labels, zeros = init_arrays(class_num,
                                                                                               sample_num_per_class,
                                                                                               batch_num_per_class)

    imgnames = os.listdir(f"./{support_dir}/label")
    testnames = os.listdir(test_dir)
    indexs = list(range(0, len(imgnames)))[0:5]
    chosen_index = indexs
    j = 0
    for k in chosen_index:
        # process image
        image_path = f"{support_dir}/image/{imgnames[k].replace('.png', '.jpg')}"
        label_path = f"{support_dir}/label/{imgnames[k]}"
        image, label = import_image_and_label(image_path, label_path)

        support_images[k] = image
        support_labels[k][0] = label

    testimage = cv2.imread(f"{test_dir}/{testname}")
    testimage = cv2.resize(testimage, (224, 224))
    testimage = normalize_and_transpose(testimage)

    query_images[0] = testimage

    class_cnt += 1
    query_images_tensor, query_labels_tensor, support_images_tensor, support_labels_tensor = transform_to_tensors(
        query_images, query_labels, support_images, support_labels, zeros)

    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor


def get_predict_batch(class_name,  class_num, sample_num_per_class,
                      batch_num_per_class, test_dir):
    class_cnt, query_images, query_labels, support_images, support_labels, zeros = init_arrays(class_num,
                                                                                               sample_num_per_class,
                                                                                               batch_num_per_class)

    imgnames = [file.split('.jpg')[0]
                for file in os.listdir(f'{test_dir}/{class_name}')
                if file.endswith('.jpg')]

    indexs = list(range(0, len(imgnames)))
    chosen_index = random.sample(indexs, sample_num_per_class + batch_num_per_class)
    j = 0
    for k in chosen_index:
        # process image

        image_path = f"{test_dir}/{class_name}/{imgnames[k]}.jpg"
        label_path = f"{test_dir}/{class_name}/{imgnames[k]}.png"
        image, label = import_image_and_label(image_path, label_path)

        if j < sample_num_per_class:
            support_images[j] = image
            support_labels[j][0] = label
        else:
            query_images[j - sample_num_per_class] = image
            query_labels[j - sample_num_per_class][0] = label
        j += 1
    query_images_tensor, query_labels_tensor, support_images_tensor, support_labels_tensor = transform_to_tensors(
        query_images, query_labels, support_images, support_labels, zeros)

    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor
