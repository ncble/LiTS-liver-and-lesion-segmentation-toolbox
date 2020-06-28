"""
There is a bug in the package medpy.io.header.Header:

ndim = len(direction.shape[0]) should be ndim = direction.shape[0]
"""

import os
import glob
import numpy as np


class Header:
    r"""
    A medpy header object.

    Stores spacing, offset/origin, direction, and possibly further meta information.
    Provide at least one of the parameters. Missing information is extracted from
    the ``sitkimage`` or, if not supplied, set to a default value. 

    Parameters
    ----------
    spacing : tuple of floats
        the image's voxel spacing
        defaults to a tuple of `1.0`s
    offset : tuple of floats
        the image's offset/origin
        defaults to a tuple of `0.0`s
    direction : ndarray
        the image's affine transformation matrix
        must be of square shape
        default to the identity matrix
    sitkimage : sitk.Image
        the simple itk image as loaded
    """

    def __init__(self, spacing=None, offset=None, direction=None, sitkimage=None):
        assert \
            sitkimage is not None or \
            spacing is not None or \
            offset is not None or \
            direction is not None

        # determin the image's ndim and default data types
        if direction is not None:
            direction = np.asarray(direction)
            ndim = direction.shape[0]
        elif offset is not None:
            offset = tuple(offset)
            ndim = len(offset)
        elif spacing is not None:
            spacing = tuple(spacing)
            ndim = len(spacing)
        else:
            ndim = len(sitkimage.GetSpacing())

        # set missing information to extracted or default values
        if spacing is None:
            spacing = sitkimage.GetSpacing() if sitkimage is not None else (1.0, ) * ndim
        if offset is None:
            offset = sitkimage.GetOrigin() if sitkimage is not None else (0.0, ) * ndim
        if direction is None:
            direction = np.asarray(sitkimage.GetDirection()).reshape(
                ndim, ndim) if sitkimage is not None else np.identity(ndim)

        # assert consistency
        assert len(spacing) == len(offset)
        assert direction.ndim == 2
        assert len(spacing) == direction.shape[0]
        assert direction.shape[0] == direction.shape[1]

        # set members
        self.spacing = spacing
        self.offset = offset
        self.direction = direction
        self.sitkimage = sitkimage

    def copy_to(self, sitkimage):
        """
        Copy all stored meta information info to an sitk Image.

        Note that only the spacing and the offset/origin information
        are guaranteed to be preserved, although the method also
        tries to copy other meta information such as DICOM tags.

        Parameters
        ----------
        sitkimage : sitk.Image
            the sitk Image object to which to copy the information

        Returns
        -------
        sitkimage : sitk.Image
            the passed sitk Image object
        """
        if self.sitkimage is not None:
            for k in self.sitkimage.GetMetaDataKeys():
                sitkimage.SetMetaData(k, self.sitkimage.GetMetaData(k))

        ndim = len(sitkimage.GetSize())
        spacing, offset, direction = self.get_info_consistent(ndim)

        sitkimage.SetSpacing(spacing)
        sitkimage.SetOrigin(offset)
        sitkimage.SetDirection(tuple(direction.flatten()))

        return sitkimage

    def get_info_consistent(self, ndim):
        """
        Returns the main meta-data information adapted to the supplied
        image dimensionality.

        It will try to resolve inconsistencies and other conflicts,
        altering the information avilable int he most plausible way.

        Parameters
        ----------
        ndim : int
            image's dimensionality

        Returns
        -------
        spacing : tuple of floats
        offset : tuple of floats
        direction : ndarray
        """
        if ndim > len(self.spacing):
            spacing = self.spacing + (1.0, ) * (ndim - len(self.spacing))
        else:
            spacing = self.spacing[:ndim]

        if ndim > len(self.offset):
            offset = self.offset + (0.0, ) * (ndim - len(self.offset))
        else:
            offset = self.offset[:ndim]

        if ndim > self.direction.shape[0]:
            direction = np.identity(ndim)
            direction[:self.direction.shape[0], :self.direction.shape[0]] = self.direction
        else:
            direction = self.direction[:ndim, :ndim]

        return spacing, offset, direction

    def set_voxel_spacing(self, spacing):
        """
        Set image's spacing.

        Parameters
        ----------
        spacing : tuple of floats
            the new image voxel spacing
            take care that image and spacing dimensionalities match
        """
        self.spacing = tuple(spacing)

    def set_offset(self, offset):
        """
        Set image's offset.

        Parameters
        ----------
        offset : tuple of floats
            the new image offset / origin
            take care that image and offset dimensionalities match
        """
        self.offset = tuple(offset)

    def set_direction(self, direction):
        """
        Set image's direction.

        Returns
        -------
        direction : tuple of floats
            the image's direction / affine transformation matrix
            must be of square shape
            default to the identity matrix
        """
        self.direction = np.asarray(direction)

    def get_voxel_spacing(self):
        """
        Get image's spacing.

        Returns
        -------
        spacing : tuple of floats
            the image's spacing
        """
        return self.spacing

    def get_offset(self):
        """
        Get image's offset.

        Returns
        -------
        offset : tuple of floats
            the image's offset / origin
        """
        return self.offset

    def get_direction(self):
        """
        Get image's direction.

        Returns
        -------
        direction : ndarray
            the image's direction / affine transformation matrix
            of square shape
        """
        return self.direction

    def get_sitkimage(self):
        """
        Get underlying sitk Image object.

        Returns
        -------
        image-object : sitk.Image or None
            the underlying sitk image object if set
        """
        return self.sitkimage


if __name__ == "__main__":
    print("Start")
