from unittest import TestCase

import numpy as np

from src.evaluation.iou import iou


class TestIoU(TestCase):

    def test_iou_should_return_1_when_masks_are_identical(self):
        # Given
        mask = np.random.randint(2, size=(2, 4))
        # When
        iou_result = iou(mask, mask)
        # Then
        expected_result = 1.0
        self.assertEqual(expected_result, iou_result)

    def test_iou_should_return_0_when_masks_are_opposites(self):
        # Given
        mask1 = np.random.randint(2, size=(2, 4))
        mask2 = np.ones((2, 4)) - mask1
        # When
        iou_result = iou(mask1, mask2)
        # Then
        expected_result = 0.0
        self.assertEqual(expected_result, iou_result)

    def test_iou_should_return_0_5_when_mask_2_is_included_in_mask_1_and_masks_have_50_percent_overlap(self):
        # Given
        mask1 = np.ones((2, 4))
        mask2 = np.concatenate((np.ones((2, 2)), np.zeros((2, 2))), axis=1)
        # When
        iou_result = iou(mask1, mask2)
        # Then
        expected_result = 0.5
        self.assertEqual(expected_result, iou_result)

    def test_iou_should_return_0_25_when_mask_2_is_included_in_mask_1_and_masks_have_25_percent_overlap(self):
        # Given
        mask1 = np.ones((2, 4))
        mask2 = np.concatenate((np.ones((2, 1)), np.zeros((2, 3))), axis=1)
        # When
        iou_result = iou(mask1, mask2)
        # Then
        expected_result = 0.25
        self.assertEqual(expected_result, iou_result)

    def test_iou_should_return_x_when_mask_2_is_not_included_in_mask_1_and_common_areas_are_50_percent_of_each_mask(
            self):
        # Given
        mask1 = np.concatenate((np.zeros((2, 1)), np.ones((2, 2)), np.zeros((2, 1))), axis=1)
        mask2 = np.concatenate((np.ones((2, 2)), np.zeros((2, 2))), axis=1)
        # When
        iou_result = iou(mask1, mask2)
        # Then
        expected_result = 1/3.0
        self.assertEqual(expected_result, iou_result)
