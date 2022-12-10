# -*- coding: utf-8 -*-

import math
import random
import pyclipper
import numpy as np
import cv2
from data_loader.augment import DataAugment

data_aug = DataAugment()


def check_and_validate_polys(polys, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

    validated_polys = []
    for poly in polys:
        p_area = cv2.contourArea(poly)
        if abs(p_area) < 1:
            continue
        validated_polys.append(poly)
    return np.array(validated_polys)

def unshrink_offset(poly,ratio):
    area = cv2.contourArea(poly)
    peri = cv2.arcLength(poly, True)
    a = 8
    b = peri - 4
    c = 1-0.5 * peri - area/ratio
    return quadratic(a,b,c)

def quadratic(a, b, c):
    if (b * b - 4 * a * c) < 0:
        return 'None'
    Delte = math.sqrt(b * b - 4 * a * c)
    if Delte > 0:
        x = (- b + Delte) / (2 * a)
        y = (- b - Delte) / (2 * a)
        return x, y
    else:
        x = (- b) / (2 * a)
        return x

def generate_rbox(im_size, text_polys, text_tags, training_mask, shrink_ratio):
    """
    生成mask图，白色部分是文本，黑色是北京
    :param im_size: 图像的h,w
    :param text_polys: 框的坐标
    :param text_tags: 标注文本框是否参与训练
    :param training_mask: 忽略标注为 DO NOT CARE 的矩阵
    :return: 生成的mask图
    """
    h, w = im_size
    G_s = np.zeros((h, w), dtype=np.float32)
    G_d = np.zeros((h, w), dtype=np.float32)
    for i, (poly, tag) in enumerate(zip(text_polys, text_tags)):
        try:
            poly = poly.astype(np.int)
            # D = cv2.contourArea(poly) * (1 - shrink_ratio * shrink_ratio) / cv2.arcLength(poly, True)
            D = cv2.contourArea(poly) * (1 - shrink_ratio) / cv2.arcLength(poly, True) + 0.5
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_poly = np.array(pco.Execute(-D))
            # dilated_poly = np.array(pco.Execute(D))
            dilated_poly = np.array(pco.Execute(0))
            cv2.fillPoly(G_s, shrinked_poly, 1)
            cv2.fillPoly(G_d, dilated_poly, 1)
            if not tag:
                cv2.fillPoly(training_mask, shrinked_poly, 0)
        except:
            print(poly)
    # return score_map, training_mask
    return G_s, G_d, training_mask


def augmentation(im: np.ndarray, text_polys: np.ndarray, scales: np.ndarray, degrees: int) -> tuple:
    # the images are rescaled with ratio {0.5, 1.0, 2.0, 3.0} randomly
    im, text_polys = data_aug.random_scale(im, text_polys, scales)
    # the images are horizontally fliped and rotated in range [−10◦, 10◦] randomly
    if random.random() < 0.5:
        im, text_polys = data_aug.horizontal_flip(im, text_polys)
    if random.random() < 0.5:
        im, text_polys = data_aug.random_rotate_img_bbox(im, text_polys, degrees)
    return im, text_polys


def image_label(im: np.ndarray, text_polys: np.ndarray, text_tags: list, input_size: int = 640,
                shrink_ratio: float = 0.4, degrees: int = 10,
                scales: tuple = (0.5, 3.0)) -> tuple:
    """
    读取图片并生成label
    :param im: 图片
    :param text_polys: 文本标注框
    :param text_tags: 是否忽略文本的标致：true 忽略, false 不忽略
    :param input_size: 输出图像的尺寸
    :param shrink_ratio: gt收缩的比例
    :param degrees: 随机旋转的角度
    :param scales: 随机缩放的尺度
    :return:
    """
    h, w, _ = im.shape
    # 检查越界
    text_polys = check_and_validate_polys(text_polys, (h, w))
    im, text_polys = augmentation(im, text_polys, scales, degrees)

    h, w, _ = im.shape
    short_edge = min(h, w)
    if short_edge < input_size:
        # 保证短边 >= inputsize
        scale = input_size / short_edge
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        text_polys *= scale

    h, w, _ = im.shape
    training_mask = np.ones((h, w), dtype=np.uint8)
    
    # G_d, training_mask = generate_rbox((h, w), text_polys, text_tags, training_mask, 1)
    
    # score_maps.append(G_d)
    # kernel = np.ones((3,3), np.uint8)
    # G_d = cv2.dilate(G_d, kernel, iterations = 1)
    # # cv2.imwrite('Gd.png', G_d * 255)
    
    G_s, G_d, training_mask = generate_rbox((h, w), text_polys, text_tags, training_mask, shrink_ratio)
    
    thres_map = G_d - G_s
    
    score_maps = [G_s, thres_map]
    
    # cv2.imwrite('th.png', thres_map * 255)
    score_maps = np.array(score_maps, dtype=np.float32)
    
    '''debug'''
    # print('====================')
    # print(score_maps.shape)
    # print(G_d.shape)
    # print(G_s.shape)
    
    imgs = data_aug.random_crop([im, score_maps.transpose((1, 2, 0)), training_mask, G_d], (input_size, input_size))

    return imgs[0], imgs[1].transpose((2, 0, 1))[:, ::4, ::4], imgs[2][::4, ::4], imgs[3][::4, ::4]  # im, score_maps, training_mask, Gd_mask#

if __name__ == '__main__':
    poly = np.array([377,117,463,117,465,130,378,130]).reshape(-1,2)
    print(poly)
    print(poly.shape) #(4,2)
    shrink_ratio = 0.5
    d_i = cv2.contourArea(poly) * (1 - shrink_ratio) / cv2.arcLength(poly, True) + 0.5
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked_poly = np.array(pco.Execute(-d_i))
    print(d_i)
    print(cv2.contourArea(shrinked_poly.astype(int)) / cv2.contourArea(poly))
    print(unshrink_offset(shrinked_poly,shrink_ratio))
