import numpy as np
import cv2
import numbers
from typing import List, Union, Sequence
from numpy.core.numeric import normalize_axis_index

interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'bilinear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

def top_k(x, k, axis=-1, largest=True, sorted=True):
    axis_size = x.size if axis is None else x.shape[axis]
    assert 1 <= k <= axis_size

    x = np.asanyarray(x)
    if largest:
        index_array = np.argpartition(x, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(x, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(x, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices

def resize_image_short(image, dst_size, return_scale=False, interpolation='bilinear'):
    assert isinstance(image, np.ndarray) and image.ndim in {2, 3}
    src_height, src_width = image.shape[:2]
    scale = max(dst_size / src_width, dst_size / src_height)
    dst_width = int(round(scale * src_width))
    dst_height = int(round(scale * src_height))

    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    x_scale = dst_width / src_width
    y_scale = dst_height / src_height
    return resized_image, x_scale, y_scale
    
def center_crop_image(image, dst_width, dst_height, strict=True):
    assert isinstance(image, np.ndarray) and image.ndim in {2, 3}
    assert isinstance(dst_width, numbers.Integral) and isinstance(dst_height, numbers.Integral)
    src_height, src_width = image.shape[:2]
    if strict:
        assert (src_height >= dst_height) and (src_width >= dst_width)

    crop_top = max((src_height - dst_height) // 2, 0)
    crop_left = max((src_width - dst_width) // 2, 0)
    return image[
        crop_top : dst_height + crop_top,
        crop_left : dst_width + crop_left,
        ...,
    ]

def normalize_image_channel(image, swap_rb=False):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif num_channels == 3:
            if swap_rb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif num_channels == 4:
            if swap_rb:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f'Unsupported image channel number, only support 1, 3 and 4, got {num_channels}!')
    else:
        raise ValueError(f'Unsupported image ndarray ndim, only support 2 and 3, got {image.ndim}!')
    return image

def normalize_image_value(image: np.ndarray, mean, std, rescale_factor=None): 
    dst_dtype = np.float32
    mean = np.array(mean, dtype=dst_dtype).flatten()
    std = np.array(std, dtype=dst_dtype).flatten()
    if rescale_factor == 'auto':
        if not np.issubdtype(image.dtype, np.unsignedinteger):
            raise TypeError(f'Only support uint dtype ndarray when `rescale_factor` is `auto`, got {image.dtype}')
        mean *= np.iinfo(image.dtype).max
        std *= np.iinfo(image.dtype).max
    elif isinstance(rescale_factor, (int, float)):
        mean *= rescale_factor
        std *= rescale_factor
    image = image.astype(dst_dtype, copy=True)
    image -= mean
    image /= std
    return image

def softmax(x: np.ndarray, axis=-1, valid_indices=None, copy=True):
    if copy:
        x = np.copy(x)
        
    if valid_indices is not None:
        interested_x = np.take(x, valid_indices, axis=axis)
    else:
        interested_x = x
        
    max_val = np.max(interested_x, axis=axis, keepdims=True)
    interested_x -= max_val
    np.exp(interested_x, interested_x)
    sum_exp = np.sum(interested_x, axis=axis, keepdims=True)
    interested_x /= sum_exp
    
    if valid_indices is not None:
        axis = normalize_axis_index(axis, x.ndim)
        x.fill(0)
        x[(slice(None),) * axis + (valid_indices,)] = interested_x
    else:
        x = interested_x
    return x

def sum_by_indices_list(x: np.ndarray, indices_list: List[List[int]] = None, axis=-1):
    axis = normalize_axis_index(axis, x.ndim)
    new_shape = list(x.shape)
    new_shape[axis] = len(indices_list)
    dst = np.empty(new_shape, dtype=x.dtype)
    for new_index, old_indices in enumerate(indices_list):
        dest_dims = (slice(None),) * axis + (new_index,)
        dst[dest_dims] = np.sum(x.take(old_indices, axis), axis=axis, keepdims=False)
    return dst

