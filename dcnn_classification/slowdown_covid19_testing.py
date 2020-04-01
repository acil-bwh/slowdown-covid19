"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import argparse
import warnings
import threading
import multiprocessing.pool

from PIL import Image as pil_image
from skimage import exposure, filters
from six.moves import range
import numpy as np
import pandas as pd

from matplotlib import image

import tensorflow as tf

from .slowdown_covid19_network import *


from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def validate_filename(filename, white_list_formats):
    """Check if a filename refers to a valid file.
    # Arguments
        filename: String, absolute path to a file
        white_list_formats: Set, allowed file extensions
    # Returns
        A boolean value indicating if the filename is valid or not
    """
    return (filename.lower().endswith(white_list_formats) and
            os.path.isfile(filename))


def save_img(path,
             x,
             data_format='channels_last',
             file_format=None,
             scale=True,
             **kwargs):
    """Saves an image stored as a Numpy array to a path or file object.
    # Arguments
        path: Path or file object.
        x: Numpy array.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        file_format: Optional file format override. If omitted, the
            format to use is determined from the filename extension.
            If a file object was used instead of a filename, this
            parameter should always be used.
        scale: Whether to rescale image values to be within `[0, 255]`.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
    """
    img = array_to_img(x, data_format=data_format, scale=scale)
    if img.mode == 'RGBA' and (file_format == 'jpg' or file_format == 'jpeg'):
        warnings.warn('The JPG format does not support '
                      'RGBA images, converting to RGB.')
        img = img.convert('RGB')
    img.save(path, format=file_format, **kwargs)


def load_img(path):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: The desired image format. One of "grayscale", "rgb", "rgba".
            "grayscale" supports 8-bit images and 32-bit signed integer images.
            Default: "rgb".
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported.
            Default: "nearest".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `load_img` requires PIL.')
    with open(path, 'rb') as f:
        img = pil_image.open(io.BytesIO(f.read()))
        if img.mode not in ('L', 'I;16', 'I'):
            img = img.convert('L')
    return img


def list_pictures(directory, ext=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif',
                                  'tiff')):
    """Lists all pictures in a directory, including all subdirectories.
    # Arguments
        directory: string, absolute path to the directory
        ext: tuple of strings or single string, extensions of the pictures
    # Returns
        a list of paths
    """
    ext = tuple('.%s' % e for e in ((ext,) if isinstance(ext, str) else ext))
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.lower().endswith(ext)]


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.
    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean, follow symbolic links to subdirectories.
    # Yields
        Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if fname.lower().endswith('.tiff'):
                warnings.warn('Using ".tiff" files with multiple bands '
                              'will cause distortion. Please verify your output.')
            if fname.lower().endswith(white_list_formats):
                yield root, fname


def _list_valid_filenames_in_directory(directory, white_list_formats, split):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean, follow symbolic links to subdirectories.
    # Returns
         classes: a list of class indices
         filenames: the path of valid files in `directory`, relative from
             `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        all_files = list(_iter_valid_files(directory, white_list_formats,
                                           False))
        num_files = len(all_files)
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = all_files[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, False)
    # classes = []
    filenames = []
    for root, fname in valid_files:
        # classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return filenames


def array_to_img(x, data_format='channels_last', scale=True, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format, either "channels_first" or "channels_last".
            Default: "channels_last".
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively.
            Default: True.
        dtype: Dtype to use.
            Default: "float32".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        if np.max(x) > 255:
            # 32-bit signed integer grayscale image. PIL mode "I"
            return pil_image.fromarray(x[:, :, 0].astype('int32'), 'I')
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))


def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


def equalize_core(img_in, clip_limit=0.01, med_filt=5, output_type='uint16'):
    if len(img_in.shape) == 3:
        img = img_in[:, :, 0]
    else:
        img = img_in.copy()

    if img.dtype is np.dtype(np.float32):
        img_norm = img / img.max()  # Format adaptation
    else:
        img_norm = img.astype('float32') / \
            np.iinfo(img.dtype).max  # Format adaptation
    img_clahe = exposure.equalize_adapthist(
        img_norm, clip_limit=clip_limit)  # CLAHE
    img_clahe_median = filters.median(img_clahe, np.ones(
        (5, 5))).astype('float32')  # Median Filter

    lower, upper = np.percentile(img_clahe_median.flatten(), [2, 98])
    img_clip = np.clip(img_clahe_median, lower, upper)
    img_out = (img_clip - lower) / (upper - lower)

    if output_type is not None:
        max_val = np.iinfo(output_type).max
        img_out = (max_val * img_out).astype(output_type)
    else:
        max_val = 1.0
    return img


def equalize(img_in, clip_limit=0.01, output_type='uint16', target_size=(1024, 1024), interpolation='nearest'):

    if len(img_in.shape) == 3:
        img = img_in[:, :, 0]
    else:
        img = img_in.copy()

    equalize_core(img, clip_limit, ouput_type=output_type)

    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img_out.shape[0] != target_size[0] or img_out.shape[1] != target_size[1]:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img_out = np.array(pil_image.fromarray(
                img_out).resize(width_height_tuple, resample))

    if len(img_in.shape) == 3:
        img_out = np.expand_dims(img_out, axis=-1)
        img_out = np.tile(img_out, (1, 1, 3))

    return img_out


class Iterator(object):
    """Base class for image data iterators.
    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """
    white_list_formats = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                # Avoiding modulo by zero error
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class BatchFromFilesMixin():
    """Adds methods related to getting batches from filenames
    It includes the logic to transform image files to batches.
    """

    def set_processing_attrs(self,
                             image_data_generator,
                             target_size,
                             data_format,
                             save_to_dir,
                             save_prefix,
                             save_format,
                             subset,
                             interpolation):
        """Sets attributes to use later for processing files into a batch.
        # Arguments
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            target_size: tuple of integers, dimensions to resize input images to.
            color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
                Color mode to read images.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
            subset: Subset of data (`"training"` or `"validation"`) if
                validation_split is set in ImageDataGenerator.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are "nearest", "bilinear", and "bicubic".
                If PIL version 1.1.3 or newer is installed, "lanczos" is also
                supported. If PIL version 3.4.0 or newer is installed, "box" and
                "hamming" are also supported. By default, "nearest" is used.
        """
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.data_format = data_format
        if self.data_format == 'channels_last':
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = (3,) + self.target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) +
                           self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = load_img(filepaths[j])
            x = img_to_array(img, data_format=self.data_format)
            x = equalize(x, target_size=self.target_size,
                         interpolation=self.interpolation)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                x = x.astype('float32')
                x = self.image_data_generator.standardize(x)

            batch_x[i] = x

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        return batch_x

    @property
    def filepaths(self):
        """List of absolute paths to image files"""
        raise NotImplementedError(
            '`filepaths` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def labels(self):
        """Class labels of every observation"""
        raise NotImplementedError(
            '`labels` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )

    @property
    def sample_weight(self):
        raise NotImplementedError(
            '`sample_weight` property method has not been implemented in {}.'
            .format(type(self).__name__)
        )


class DirectoryIterator(BatchFromFilesMixin, Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: string, path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
            If set to False, sorts the data in alphanumeric order.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        follow_links: boolean,follow symbolic links to subdirectories
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        dtype: Dtype to use for generated arrays.
    """
    allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        super(DirectoryIterator, self).set_processing_attrs(image_data_generator,
                                                            target_size,
                                                            data_format,
                                                            save_to_dir,
                                                            save_prefix,
                                                            save_format,
                                                            subset,
                                                            interpolation)
        self.directory = directory
        self.dtype = dtype
        self.samples = 0

        pool = multiprocessing.pool.ThreadPool()

        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        # for dirpath in directory:
        results.append(
            pool.apply_async(_list_valid_filenames_in_directory,
                             (directory, self.white_list_formats, self.split)))
        for res in results:
            filenames = res.get()
            self.filenames += filenames
        self.samples = len(self.filenames)

        print('Found %d images.' % (self.samples))
        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]
        super(DirectoryIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return None

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None


class ImageDataGenerator(object):
    def __init__(self, rescale=None, data_format='channels_last', validation_split=0.0, interpolation_order=1,
                 dtype='float32'):

        self.rescale = rescale
        self.dtype = dtype
        self.interpolation_order = interpolation_order

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                '(channel after row and column) or '
                '`"channels_first"` (channel before row and column). '
                'Received: %s' % data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                '`validation_split` must be strictly between 0 and 1. '
                ' Received: %s' % validation_split)
        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.principal_components = None

    def flow_from_directory(self,
                            directory,
                            target_size=(256, 256),
                            batch_size=32,
                            interpolation='nearest'):
        """Takes the path to a directory & generates batches of augmented data.
        # Arguments
            directory: string, path to the target directory.
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree
                will be included in the generator.
                See [this script](
                https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
                for more details.
            target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to
                have 1, 3, or 4 channels.
            batch_size: Size of the batches of data (default: 32).
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.
        # Returns
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a NumPy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a NumPy array of corresponding labels.
        """
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=False,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            subset=None,
            interpolation=interpolation,
            dtype=self.dtype
        )

    def standardize(self, x):
        """Applies the normalization configuration in-place to a batch of inputs.
        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network. If a copy of `x`
        would be created instead it would have a significant performance cost.
        If you want to apply this method without changing the input in-place
        you can call the method creating a copy before:
        standardize(np.copy(x))
        # Arguments
            x: Batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.rescale:
            x *= self.rescale
        return x


def test(in_dir, model_path, out_csv_file, batch_size=32):
    # network_model = load_model(model_path)
    p = dict()
    p['target_image_size'] = [1024, 1024]
    p['use_imagenet_weights'] = False
    p['net_type'] = 'TBNet'

    network = SlowdownCOVID19Network(**p)
    network.build_model(True, optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                        loss_function='categorical_crossentropy', additional_metrics=['acc'],
                        pretrained_weights_file_path=model_path)
    network_model = network.model

    datagen = ImageDataGenerator(
        rescale=1/65536., validation_split=0.0, dtype='float32')

    test_generator = datagen.flow_from_directory(in_dir,
                                                 target_size=(1024, 1024),
                                                 batch_size=batch_size)

    tot_num_images = test_generator.samples
    results = {'normal': np.zeros((tot_num_images,)), 'mild': np.zeros((tot_num_images,)),
               'moderate-severe': np.zeros((tot_num_images,))}

    "Predicting probabilities..."
    for ii in range(0, tot_num_images, batch_size):
        img_batch = test_generator.next()
        batch_predictions = network_model.predict(img_batch)

        results['normal'][ii:ii+batch_size] = batch_predictions[:, 0]
        results['mild'][ii:ii+batch_size] = batch_predictions[:, 1]
        results['moderate-severe'][ii:ii+batch_size] = batch_predictions[:, 2]

    df = pd.DataFrame.from_dict(results)
    df.to_csv(out_csv_file)


def test2(in_dir, model_path, out_csv_file, batch_size=32):

    file_list = [join(in_dir, f)
                 for f in listdir(in_dir) if isfile(join(in_dir, f))]

    image_list = list()
    case_list = list()
    for ff in file_list:
        try:
            img = image.imread(ff)
            img2 = equalize_core(img, output_type='uint16')
            img2 = resize(img2, (1024, 1024),anti_aliasing=True)
            img2 = gray2rgb(img2)
            img2=img2.astype('float32')/65536.
            image_list.append(img2)
            case_list.append(ff)
        except:
            print("Could not read image")

    X = np.concatenate(image_list, axis=0)
    p = dict()
    p['target_image_size'] = [1024, 1024]
    p['use_imagenet_weights'] = False
    p['net_type'] = 'TBNet'
    network = SlowdownCOVID19Network(**p)
    network.build_model(True, optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                        loss_function='categorical_crossentropy', additional_metrics=['acc'],
                        pretrained_weights_file_path=model_path)
    network_model = network.model
    
    preds = network_model.predict(X, batch_size=batch_size)
    
    results = dict()
    results['case'] = case_list
    results['normal'] = preds[:, 0]
    results['mild'] = preds[:, 1]
    results['moderate-severe'] = preds[:, 2]
    
    df = pd.DataFrame.from_dict(results)
    
    df.to_csv(out_csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test Slowdown COVID-19 models.')
    parser.add_argument(
        '--i', help="Input directory with images", type=str, required=True)
    parser.add_argument('--m', help="DNN model path", type=str, required=True)
    parser.add_argument(
        '--o', help="Output csv file to save results", type=str, required=True)
    parser.add_argument('--bs', help="Batch size to use. Default is 32. Recommended value is a multiple of the number "
                                     "of testing images", type=int, required=False, default=32)
    args = parser.parse_args()
    test(args.i, args.m, args.o, int(args.bs))
