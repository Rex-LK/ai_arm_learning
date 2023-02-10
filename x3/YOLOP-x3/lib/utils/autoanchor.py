# Auto-anchor utils

import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm
from lib.utils import is_parallel


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def run_anchor(logger,dataset, model, thr=4.0, imgsz=640):
    det = model.module.model[model.module.detector_index] if is_parallel(model) \
        else model.model[model.detector_index]
    anchor_num = det.na * det.nl
    new_anchors = kmean_anchors(dataset, n=anchor_num, img_size=imgsz, thr=thr, gen=1000, verbose=False)
    new_anchors = torch.tensor(new_anchors, device=det.anchors.device).type_as(det.anchors)
    det.anchor_grid[:] = new_anchors.clone().view_as(det.anchor_grid)  # for inference
    det.anchors[:] = new_anchors.clone().view_as(det.anchors) / det.stride.to(det.anchors.device).view(-1, 1, 1)  # loss
    check_anchor_order(det)
    logger.info(str(det.anchors))
    print('New anchors saved to model. Update model config to use these anchors in the future.')


def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(path, str):  # not class
        raise TypeError('Dataset must be class, but found str')
    else:
        dataset = path  # dataset

    labels = [db['label'] for db in dataset.db]
    labels = np.vstack(labels)
    if not (labels[:, 1:] <= 1).all():
        # normalize label
        labels[:, [2, 4]] /= dataset.shapes[0]
        labels[:, [1, 3]] /= dataset.shapes[1]
    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max()
    # wh0 = np.concatenate([l[:, 3:5] * shapes for l in labels])  # wh
    wh0 = labels[:, 3:5] * shapes
    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)
