"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0

data augmentation with numpy and opencv for pytorch data-loader

Pytorch build-in data augmentation used PIL.Image, however some functionalities are not
supported. For instance, 

    - transforms.ColorJitter doesn't work with ToPILImage(mode='F') # one-channel image
    - transforms.

"""
import numpy as np
import cv2
import torch


class OpenCVdummy(object):
    """
    [Usage]:

    Warning there are two type of transformation
        1. geometry_transform: rotation, warpAffine, translation, etc
        2. pixelwise_transform: color jitter, brightness, saturation, ToTensor(), normalize() etc

    For (1.), we need both the image and the mask
    For (2.), only need the image

    """

    @staticmethod
    def _is_numpy_image(img):
        return isinstance(img, np.ndarray)


class OpenCVRotation(OpenCVdummy):
    """Counter-clockwise (random) rotation for torchvision.transform
    Support 2D images of shape (H, W, 3) or (H, W) or (H, W, 1)

    Random rotation/rescale:
        - angles (tuple/list): in degree
        - scales (tuple/list): resize factor 
        - centers (tuple/list): percentage of translation to the center
    """

    def __init__(self, angles=(0., 0.), scales=(1.0, 1.0), centers=(0, 0)):

        assert len(angles) == 2, \
            'angles should be a list/tuple of (lower, upper) in degree'

        self.angles = angles
        self.scales = scales
        self.centers = centers

    def __call__(self, img, msk):
        # np_rand_state = np.random.RandomState()
        # seed = np.random.get_state()[1][0]
        # np.random.seed(seed)
        # angle = np.float32(np_rand_state.uniform(*self.angles))
        scale = np.random.uniform(*self.scales)
        angle = np.float32(np.random.uniform(*self.angles))
        translate = np.random.uniform(*self.centers)

        # check type of img
        if not self._is_numpy_image(img):
            raise TypeError('img should be numpy array. Got {}'.format(type(img)))

        if len(img.shape) == 3:

            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            elif img.shape[-1] == 3:
                pass
            else:
                raise TypeError("")
        elif len(img.shape) == 2:
            pass
        else:
            raise ValueError("Image shape should be (H, W, 3) or (H, W)")

        # This following line is a shit code, not mine.
        # for channel ...
        # channel = skimage.transform.rotate(channel, self.angles, resize=False, order=1, preserve_range=True)

        # use opencv to rotate image
        h, w = img.shape[:2]  # (H, W, C)
        
        center = (int(w/2)+int(translate*w), int(h/2)+int(translate*h))
        
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h))
        rotated_msk = cv2.warpAffine(msk, M, (w, h))

        return rotated.astype(np.float32), rotated_msk.astype(int)

    def debug(self):
        from matplotlib import pyplot as plt
        img = np.zeros((50, 50, 3))
        img[:, 3:6, ...] = 1

        rot = self(img)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img)
        axarr[1].imshow(rot)
        plt.show()




if __name__ == "__main__":
    print("Start")
    A = OpenCVRotation((43, 45))
    # img = cv2.imread("../../demo/2.jpg")
    A.debug()
    # import ipdb; ipdb.set_trace()
