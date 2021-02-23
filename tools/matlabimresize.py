"""
OpenCV and PyTorch using A=-0.75 in bicubic while matlab and PIL.Image.resize using A=-0.5
https://github.com/fatheral/matlab_imresize
baseline: (factor: 4)
down scale | up scale | dataset | psnr | ssim |
matlab | matlab | Set14 | 26.05 | 0.704 |
matlab | OpenCV | Set14 | 25.11 | 0.711 |
OpenCV | matlab | Set14 | 26.17 | 0.708 |
OpenCV | OpenCV | Set14 | 24.97 | 0.705 |
PIL | matlab | Set14 | 25.75 | 0.695 |
"""
from math import ceil

import numpy as np


def derive_size_from_scale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(ceil(scale[k] * img_shape[k]))
    return output_shape


def derive_scale_from_size(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def triangle(x):
    x = np.array(x).astype(np.float32)
    lessthanzero = np.logical_and(x >= -1, x < 0)
    greaterthanzero = np.logical_and(x <= 1, x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f


def cubic(x, a):
    x = np.array(x).astype(np.float32)
    abs_x = np.abs(x)
    abs_x2 = abs_x ** 2
    abs_x3 = abs_x * abs_x2
    f = np.multiply((a + 2) * abs_x3 - (a + 3) * abs_x2 + 1, abs_x <= 1) + np.multiply(
        a * abs_x3 - 5 * a * abs_x2 + 8 * a * abs_x - 4 * a,
        (1 < abs_x) & (abs_x <= 2))
    return f


def down_kernel(kernel, scale):
    def new_kernel(x, a):
        return scale * kernel(scale * x, a)

    return new_kernel


def contributions(in_length, out_length, scale, kernel, k_width, a):
    if scale < 1:
        h = down_kernel(kernel, scale)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = ceil(kernel_width) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1, a=a)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    else:
        raise NotImplementedError(f'dim should be 0 or 1, get {dim}')
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def resize_along_dim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(image, scale=None, method='bicubic', shape=None, mode="vec", a=-0.5):
    if method is 'bicubic':
        kernel = cubic
    elif method is 'bilinear':
        kernel = triangle
    else:
        raise ValueError('Unidentified method supplied')

    kernel_width = 4.0
    # Fill scale and output_size
    if scale and shape is not None:
        raise ValueError('Only one of arguments:{scale, shape} can be assigned!')
    if scale:
        scale = float(scale)
        scale = [scale, scale]
        output_size = derive_size_from_scale(image.shape, scale)
    elif shape is not None:
        scale = derive_scale_from_size(image.shape, shape)
        output_size = list(shape)
    else:
        print('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(image.shape[k], output_size[k], scale[k], kernel, kernel_width, a=a)
        weights.append(w)
        indices.append(ind)
    dst_image = np.copy(image)
    float_2d = False
    if dst_image.ndim == 2:
        dst_image = np.expand_dims(dst_image, axis=2)
        float_2d = True
    for k in range(2):
        dim = order[k]
        dst_image = resize_along_dim(dst_image, dim, weights[dim], indices[dim], mode)
    if float_2d:
        dst_image = np.squeeze(dst_image, axis=2)
    return dst_image.astype(image.dtype)


def convert_double_to_byte(src):
    dst = np.clip(src, 0.0, 1.0)
    dst = 255 * dst
    return np.around(dst).astype(np.uint8)


if __name__ == '__main__':
    img = np.random.rand(32, 32, 1)
    imresize(img, scale=0.5)
    img = np.random.rand(32, 32, 3)
    imresize(img, scale=1.2, a=-1.)
