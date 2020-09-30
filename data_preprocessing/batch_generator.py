import os
import random

import cv2

from data_preprocessing.transform_init_tensors import init_arrays, normalize_and_transpose, import_image_and_label, transform_to_tensors


def get_training_batch(class_num, sample_num_per_class, batch_num_per_class):  # shuffle in query_images not done
    # classes.remove(EXCLUDE_CLASS)
    classes_name = os.listdir('./fewshot/support/')
    classes = list(range(0, len(classes_name)))

    chosen_classes = random.sample(classes, class_num)
    class_cnt, query_images, query_labels, support_images, support_labels, zeros = init_arrays(class_num,
                                                                                               sample_num_per_class,
                                                                                               batch_num_per_class)
    for i in chosen_classes:
        # print ('class %s is chosen' % i)
        imgnames = os.listdir('./fewshot/support/%s/label' % classes_name[i])
        indexs = list(range(0, len(imgnames)))
        chosen_index = random.sample(indexs, sample_num_per_class + batch_num_per_class)
        j = 0
        for k in chosen_index:
            # process image
            image_path = './fewshot/support/%s/image/%s' % (classes_name[i], imgnames[k].replace('.png', '.jpg'))
            label_path = './fewshot/support/%s/label/%s' % (classes_name[i], imgnames[k])
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
                        batch_num_per_class, support_dir, test_dir):  # shuffle in query_images not done
    # classes_name = os.listdir('./%s' % args.support_dir)
    # classes_name = ['android_robot', 'bucket_water' , 'nintendo_gameboy']

    class_cnt, query_images, query_labels, support_images, support_labels, zeros = init_arrays(class_num,
                                                                                               sample_num_per_class,
                                                                                               batch_num_per_class)

    # print ('class %s is chosen' % i)
    # classnames = ['english_foxhound', 'guitar']
    imgnames = os.listdir('./%s/label' % support_dir)
    # print (args.support_dir, imgnames)
    testnames = os.listdir('%s' % test_dir)
    indexs = list(range(0, len(imgnames)))[0:5]
    chosen_index = indexs
    j = 0
    for k in chosen_index:
        # process image
        image_path = '%s/image/%s' % (support_dir, imgnames[k].replace('.png', '.jpg'))
        label_path = '%s/label/%s' % (support_dir, imgnames[k])
        image, label = import_image_and_label(image_path, label_path)

        support_images[k] = image
        support_labels[k][0] = label

    testimage = cv2.imread('%s/%s' % (test_dir, testname))
    testimage = cv2.resize(testimage, (224, 224))
    testimage = normalize_and_transpose(testimage)

    query_images[0] = testimage

    class_cnt += 1
    query_images_tensor, query_labels_tensor, support_images_tensor, support_labels_tensor = transform_to_tensors(
        query_images, query_labels, support_images, support_labels, zeros)

    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor
