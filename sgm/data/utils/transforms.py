import torch
import numpy as np
from scipy import ndimage
import torch
import numpy as np


class Resize_image(object):
    """
    Resize_image is a class that represents an image resizing transformation.

    Args:
        size (tuple): The desired size of the image after resizing. Default is (3, 256, 256).
        f16 (bool): Whether to use float16 data type for the size. Default is False.

    Raises:
        ValueError: If each dimension of size is not defined.

    Attributes:
        size (ndarray): The size of the image after resizing.
        f16 (bool): Whether float16 data type is used for the size.

    """
    def __init__(self, size=(3, 256, 256), f16=False):
        if not _isArrayLike(size):
            raise ValueError('each dimension of size must be defined')
        if f16:
            self.size = np.array(size, dtype=np.float16)
        else:
            self.size = np.array(size, dtype=np.float32)
        self.f16 = f16

    def __call__(self, img):
        """
        Resize the input image to the specified size.

        Args:
            img (ndarray): The input image to be resized.

        Returns:
            ndarray: The resized image.

        Raises:
            AssertionError: If the input image has dtype f16.

        """
        z, x, y = img.shape
        assert not self.f16, "Resize_image not supported for f16"
        ori_shape = np.array((z, x, y), dtype=np.float32)
        resize_factor = self.size / ori_shape
        return ndimage.interpolation.zoom(img, resize_factor, order=1)


class Normalization(object):
    """
    A class that performs normalization on input images.

    Args:
        min (float): The minimum value of the input range.
        max (float): The maximum value of the input range.
        f16 (bool, optional): Whether to use float16 data type. Defaults to False.
        round_v (int, optional): The number of decimal places to round the output to. Defaults to 6.
    """
    def __init__(self, min, max, f16=False, round_v=6):
        dtype = torch.float16 if f16 else torch.float32
        self.range = torch.tensor([min, max], dtype=dtype)
        self.round_v = round_v

    def __call__(self, img: torch.Tensor):
        """
        Normalize the input image.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The normalized image tensor.
        """
        return torch.round((img - self.range[0]) / (self.range[1] - self.range[0]), decimals=self.round_v)



class Normalization_min_max(object):
    """
    Applies min-max normalization to an image.

    Args:
        min_v (float): The minimum value for normalization.
        max_v (float): The maximum value for normalization.
        eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-4.
        remove_noise (bool, optional): Whether to remove noise from the image. Removes all HU values smaller than 200. Defaults to True.
    """
    def __init__(self, min_v, max_v, eps=1e-4, remove_noise=True):
        self.max = max_v
        self.min = min_v
        self.eps = eps
        self.remove_noise = remove_noise

    def __call__(self, img):
        """
        Applies the transformation to the input image.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The transformed image.
        """
        # Removing noise from xray machine
        if self.remove_noise:
            img[img < 200] = 0
        img_min = np.min(img)
        img_max = np.max(img)
        img_out = (self.max - self.min) * (img - img_min) / (img_max - img_min + self.eps) + self.min
        return img_out


class ReturnIdentity(object):
    """
    A transformation class that returns the input image as it is.
    """

    def __call__(self, img):

        return img


class ToTensor(object):
    """
    Converts a PIL Image or numpy array to a PyTorch tensor.

    Args:
        f16 (bool, optional): Whether to convert the tensor to float16 dtype. Default is False.

    Returns:
        torch.Tensor: Converted PyTorch tensor.

    """
    def __init__(self, f16=False):
        self.f16 = f16

    def __call__(self, img):
        dtype = np.float16 if self.f16 else np.float32
        return torch.from_numpy(np.array(img).astype(dtype))

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
