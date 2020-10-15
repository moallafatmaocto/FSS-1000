from unittest import TestCase

import torch
from pandas import np

from src.data_preprocessing.transform_init_tensors import init_arrays, normalize_and_transpose


class TestInitTensors(TestCase):
    def test_init_arrays_returns_the_right_unitary_size_nparrays(self):
        # Given
        class_num = 1
        sample_num_per_class = 1
        batch_num_per_class = 1

        # When
        _, query_images, query_labels, support_images, support_labels, zeros = init_arrays(
            class_num, sample_num_per_class, batch_num_per_class)
        # Then

        expected_query_images = np.zeros((1, 3, 224, 224))
        expected_query_labels = np.zeros((1, 1, 224, 224))
        expected_support_images = np.zeros((1, 3, 224, 224))
        expected_support_labels = np.zeros((1, 1, 224, 224))
        zeros = np.zeros((1, 1, 224, 224))

        self.assertEqual(np.shape(expected_query_images), np.shape(query_images))
        self.assertEqual(np.shape(expected_query_labels), np.shape(query_labels))
        self.assertEqual(np.shape(expected_support_images), np.shape(support_images))
        self.assertEqual(np.shape(expected_support_labels), np.shape(support_labels))

    def test_init_arrays_returns_the_right_size_for_nparrays(self):
        # Given
        class_num = 1
        sample_num_per_class = 5
        batch_num_per_class = 2

        # When
        _, query_images, query_labels, support_images, support_labels, zeros = init_arrays(
            class_num, sample_num_per_class, batch_num_per_class)
        # Then

        expected_query_images = np.zeros((2, 3, 224, 224))
        expected_query_labels = np.zeros((2, 1, 224, 224))
        expected_support_images = np.zeros((5, 3, 224, 224))
        expected_support_labels = np.zeros((5, 1, 224, 224))
        zeros = np.zeros((2, 1, 224, 224))

        self.assertEqual(np.shape(expected_query_images), np.shape(query_images))
        self.assertEqual(np.shape(expected_query_labels), np.shape(query_labels))
        self.assertEqual(np.shape(expected_support_images), np.shape(support_images))
        self.assertEqual(np.shape(expected_support_labels), np.shape(support_labels))

    def test_normalize_and_transpose_returns_the_right_shape_and_colors(self):
        # Given
        old_image = np.ones((2, 5, 3)) * [5, 200, 110]  # rgb
        # When
        image = normalize_and_transpose(old_image)
        # Then
        expected_image_size = np.shape(np.ones((3, 2, 5)))
        expected_image_colors = [110 / 255., 200 / 255., 5 / 255.]
        self.assertEqual(expected_image_size, np.shape(image))
        self.assertEqual(expected_image_colors[0], image[0, 0, 0])  # b
        self.assertEqual(expected_image_colors[1], image[1, 0, 0])  # g
        self.assertEqual(expected_image_colors[2], image[2, 0, 0])  # r
