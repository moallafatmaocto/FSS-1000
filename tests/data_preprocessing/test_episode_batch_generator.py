from unittest import TestCase

import torch

from src.data_preprocessing.batch_generator import get_autolabel_batch


class TestBatchGenerator(TestCase):

    def test_get_autolabel_batch_returns_the_right_size_for_support_and_query_batch(self):
        # Given
        testname = '1.jpg'
        class_num = 1
        sample_num_per_class = 5
        batch_num_per_class = 1
        support_dir = 'test_data/african_elephant/supp'
        test_dir = 'test_data/african_elephant/test'

        # When
        support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor = get_autolabel_batch(
            testname, class_num, sample_num_per_class, batch_num_per_class, support_dir, test_dir
        )

        # Then
        self.assertEqual(torch.Size([5, 4, 224, 224]), support_images_tensor.size())
        self.assertEqual(torch.Size([5, 1, 224, 224]), support_labels_tensor.size())
        self.assertEqual(torch.Size([1, 4, 224, 224]), query_images_tensor.size())
        self.assertEqual(torch.Size([1, 1, 224, 224]), query_labels_tensor.size())
