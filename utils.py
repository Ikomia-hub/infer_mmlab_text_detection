import numpy as np
from scipy.linalg import det

textdet_models = {
            'DB_r18': {
                'config':
                'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
            },
            'DB_r50': {
                'config':
                'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20210325-91cef9af.pth'
            },
            'DRRG': {
                'config': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt': 'drrg/drrg_r50_fpn_unet_1200e_ctw1500-1abf4f67.pth'
            },
            'FCE_IC15': {
                'config': 'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
                'ckpt': 'fcenet/fcenet_r50_fpn_1500e_icdar2015-d435c061.pth'
            },
            'FCE_CTW_DCNv2': {
                'config': 'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
                'ckpt': 'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500-05d740bb.pth'
            },
            'MaskRCNN_CTW': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
            },
            'MaskRCNN_IC15': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
            },
            'MaskRCNN_IC17': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
            },
            'PANet_CTW': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
            },
            'PANet_IC15': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
            },
            'PS_CTW': {
                'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
            },
            'PS_IC15': {
                'config':
                'psenet/psenet_r50_fpnf_600e_icdar2015.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
            },
            'TextSnake': {
                'config':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
            }
        }


def area(pts):
    """
    :param pts: array of coordinates of vertices [[x,y],[x,y]...]
    :return: area of quadrilateral or triangle formed by the points
    """
    n = len(pts)
    if n == 4:
        return area(pts[:3]) + area(np.roll(pts, [-2], axis=[0])[:3])
    if n == 3:
        mat = np.ones((3, 3))
        mat[:, 0] = pts[:, 0]
        mat[:, 1] = pts[:, 1]
        return round(abs(0.5 * det(mat)), 1)