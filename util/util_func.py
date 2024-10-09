import SimpleITK as sitk
import torch
import numpy as np
import math
from skimage.metrics import structural_similarity

def tensor2nii(tensor, dst_nii, spacing=None, origin=None):
    img = tensor.cpu().detach().numpy()
    image = sitk.GetImageFromArray(img)
    if spacing is not None:
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
    sitk.WriteImage(image, dst_nii)

def data_norm(x):
    # self normalization
    return (x-x.min())/(x.max()-x.min())

def get_psnr(pred, target):
    """
    Compute PSNR of two tensors (2D/3D) in decibels.
    pred/target should be of same size or broadcastable
    The max intensity should be 1, thus, it's better
    to normalize into [0,1]
    """

    mse = ((pred - target) ** 2).mean()
    if mse!=0:
      psnr = -10 * math.log10(mse)
    else:
      psnr = 'INF'
    return psnr

def get_ssim_2d(pred, target, data_range=1.0):
    """
    Compute SSIM of two tensors (2D) in decibels.
    pred/target should be of same size or broadcastable
    The max intensity should be 1, thus, it's better
    to normalize into [0,1]
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    ssim = structural_similarity(pred, target, data_range=data_range)
    return ssim

def get_ssim_3d(arr1, arr2, size_average=True, data_range=None):
    '''
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    '''
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = structural_similarity(arr1_d[i], arr2_d[i], data_range=data_range)
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)

    # Height
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = []
    for i in range(N):
        ssim = structural_similarity(arr1_h[i], arr2_h[i], data_range=data_range)
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)

    # Width
    ssim_w = []
    for i in range(N):
        ssim = structural_similarity(arr1[i], arr2[i], data_range=data_range)
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return ssim_avg.mean()
    else:
        return ssim_avg

def fmt_loss_str(losses):
    return (" " + " ".join(k + ":" + str(losses[k]) for k in losses))