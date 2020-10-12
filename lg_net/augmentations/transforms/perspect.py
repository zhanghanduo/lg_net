from math import floor, ceil
import random
import numpy as np
import cv2
from PIL import Image


class Skew:
    """
        This class is used to perform perspective skewing on images
         (together with mask and keypoints).
    """

    def __init__(self, skew_type="TILT_LEFT_RIGHT", magnitude=1.0):
        super().__init__()
        self.skew_type = skew_type
        self.magnitude = magnitude

    def perform_operation(self, images, masks, keypoints=None):
        w, h = images[0].size

        x1 = 0
        x2 = h
        y1 = 0
        y2 = w

        original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]

        max_skew_amount = max(w, h)
        max_skew_amount = int(ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        if self.skew_type == "RANDOM":
            skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
        else:
            skew = self.skew_type

        # We have two choices now: we tilt in one of four directions
        # or we skew a corner.
        if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":

            if skew == "TILT":
                skew_direction = random.randint(0, 3)
            elif skew == "TILT_LEFT_RIGHT":
                skew_direction = random.randint(0, 1)
            elif skew == "TILT_TOP_BOTTOM":
                skew_direction = random.randint(2, 3)

            if skew_direction == 0:
                # Left Tilt
                new_plane = [(y1, x1 - skew_amount),  # Top Left
                             (y2, x1),  # Top Right
                             (y2, x2),  # Bottom Right
                             (y1, x2 + skew_amount)]  # Bottom Left
            elif skew_direction == 1:
                # Right Tilt
                new_plane = [(y1, x1),  # Top Left
                             (y2, x1 - skew_amount),  # Top Right
                             (y2, x2 + skew_amount),  # Bottom Right
                             (y1, x2)]  # Bottom Left
            elif skew_direction == 2:
                # Forward Tilt
                new_plane = [(y1 - skew_amount, x1),  # Top Left
                             (y2 + skew_amount, x1),  # Top Right
                             (y2, x2),  # Bottom Right
                             (y1, x2)]  # Bottom Left
            elif skew_direction == 3:
                # Backward Tilt
                new_plane = [(y1, x1),  # Top Left
                             (y2, x1),  # Top Right
                             (y2 + skew_amount, x2),  # Bottom Right
                             (y1 - skew_amount, x2)]  # Bottom Left

        if skew == "CORNER":

            skew_direction = random.randint(0, 7)

            if skew_direction == 0:
                # Skew possibility 0
                new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 1:
                # Skew possibility 1
                new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 2:
                # Skew possibility 2
                new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
            elif skew_direction == 3:
                # Skew possibility 3
                new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
            elif skew_direction == 4:
                # Skew possibility 4
                new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
            elif skew_direction == 5:
                # Skew possibility 5
                new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
            elif skew_direction == 6:
                # Skew possibility 6
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
            elif skew_direction == 7:
                # Skew possibility 7
                new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]

        # To calculate the coefficients required by PIL for the perspective skew,
        # see the following Stack Overflow discussion: https://goo.gl/sSgJdj
        matrix = []

        for p1, p2 in zip(new_plane, original_plane):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        m1 = np.array(matrix, dtype=np.float)
        m2 = np.array(original_plane).reshape(8)

        perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(m1), m2)
        perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

        def do(image):
            return image.transform(image.size,
                                   Image.PERSPECTIVE,
                                   perspective_skew_coefficients_matrix,
                                   resample=Image.BICUBIC)

        augmented_images = []
        augmented_masks = []
        if masks:
            for image, mask in zip(images, masks):
                augmented_images.append(do(image))
                augmented_masks.append(do(mask))
            return augmented_images, augmented_masks
        else:
            for image in images:
                augmented_images.append(do(image))
            return augmented_images


class Skew_cv2:
    """
        This class is used to perform perspective skewing on images
         (together with mask and keypoints).
    """

    def __init__(self, skew_type="TILT_LEFT_RIGHT", magnitude=1.0):
        super().__init__()
        self.skew_type = skew_type
        self.magnitude = magnitude

    def perform_operation(self, images, masks, keypoints=None):
        dim = images[0].shape
        w = dim[1]
        h = dim[0]

        x1 = 0
        x2 = h
        y1 = 0
        y2 = w

        original_plane = np.array([[0, 0],
                                   [w, 0],
                                   [w, h],
                                   [0, h]], dtype=np.float32)
        max_skew_amount = max(w, h)
        max_skew_amount = int(ceil(max_skew_amount * self.magnitude))
        skew_amount = random.randint(1, max_skew_amount)

        if self.skew_type == "RANDOM":
            skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
        else:
            skew = self.skew_type

        # We have two choices now: we tilt in one of four directions
        # or we skew a corner.
        if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":

            if skew == "TILT":
                skew_direction = random.randint(0, 3)
            elif skew == "TILT_LEFT_RIGHT":
                skew_direction = random.randint(0, 1)
            elif skew == "TILT_TOP_BOTTOM":
                skew_direction = random.randint(2, 3)

            if skew_direction == 0:
                # Left Tilt
                new_plane = np.array([[y1, x1 - skew_amount],  # Top Left
                                      [y2, x1],  # Top Right
                                      [y2, x2],  # Bottom Right
                                      [y1, x2 + skew_amount]])  # Bottom Left
            elif skew_direction == 1:
                # Right Tilt
                new_plane = np.array([[y1, x1],  # Top Left
                                     [y2, x1 - skew_amount],  # Top Right
                                     [y2, x2 + skew_amount],  # Bottom Right
                                     [y1, x2]]) # Bottom Left
            elif skew_direction == 2:
                # Forward Tilt
                new_plane = np.array([(y1 - skew_amount, x1),  # Top Left
                                     (y2 + skew_amount, x1),  # Top Right
                                     (y2, x2),  # Bottom Right
                                     (y1, x2)])  # Bottom Left
            elif skew_direction == 3:
                # Backward Tilt
                new_plane = np.array([(y1, x1),  # Top Left
                                     (y2, x1),  # Top Right
                                     (y2 + skew_amount, x2),  # Bottom Right
                                     (y1 - skew_amount, x2)])  # Bottom Left

        if skew == "CORNER":

            skew_direction = random.randint(0, 7)

            if skew_direction == 0:
                # Skew possibility 0
                new_plane = np.array([(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)])
            elif skew_direction == 1:
                # Skew possibility 1
                new_plane = np.array([(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)])
            elif skew_direction == 2:
                # Skew possibility 2
                new_plane = np.array([(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)])
            elif skew_direction == 3:
                # Skew possibility 3
                new_plane = np.array([(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)])
            elif skew_direction == 4:
                # Skew possibility 4
                new_plane = np.array([(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)])
            elif skew_direction == 5:
                # Skew possibility 5
                new_plane = np.array([(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)])
            elif skew_direction == 6:
                # Skew possibility 6
                new_plane = np.array([(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)])
            elif skew_direction == 7:
                # Skew possibility 7
                new_plane = np.array([(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)])

        perspective_matrix = cv2.getPerspectiveTransform(
            original_plane, new_plane.astype(np.float32))

        def do(image):
            return cv2.warpPerspective(image, perspective_matrix, (w, h))

        augmented_images = []
        augmented_masks = []
        if masks:
            for image, mask in zip(images, masks):
                augmented_images.append(do(image))
                augmented_masks.append(do(mask))
            return augmented_images, augmented_masks
        else:
            for image in images:
                augmented_images.append(do(image))
            return augmented_images
