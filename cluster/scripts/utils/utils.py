from numba import jit
import numpy as np


@jit
def pixel_sharing_bipartite(lab1, lab2):
    assert lab1.shape == lab2.shape
    psg = np.zeros((lab1.max()+1, lab2.max()+1), dtype=np.int)
    for i in range(lab1.size):
        psg[lab1.flat[i], lab2.flat[i]] += 1
    return psg

def intersection_over_union(psg):
    rsum = np.sum(psg, 0, keepdims=True)
    csum = np.sum(psg, 1, keepdims=True)
    return psg / (rsum + csum - psg)

def matching_overlap(psg, fractions=(0.5,0.5)):
    """
    create a matching given pixel_sharing_bipartite of two label images based on mutually overlapping regions of sufficient size.
    NOTE: a true matching is only gauranteed for fractions > 0.5. Otherwise some cells might have deg=2 or more.
    NOTE: doesnt break when the fraction of pixels matching is a ratio only slightly great than 0.5? (but rounds to 0.5 with float64?)
    """
    afrac, bfrac = fractions
    tmp = np.sum(psg, axis=1, keepdims=True)
    m0 = np.where(tmp==0,0,psg / tmp)
    tmp = np.sum(psg, axis=0, keepdims=True)
    m1 = np.where(tmp==0,0,psg / tmp)
    m0 = m0 > afrac
    m1 = m1 > bfrac
    matching = m0 * m1
    matching = matching.astype('bool')
    return matching

def seg(lab_gt, lab, partial_dataset=False):
    """
    calculate seg from pixel_sharing_bipartite
    seg is the average conditional-iou across ground truth cells
    conditional-iou gives zero if not in matching
    ----
    calculate conditional intersection over union (CIoU) from matching & pixel_sharing_bipartite
    for a fraction > 0.5 matching. Any CIoU between matching pairs will be > 1/3. But there may be some
    IoU as low as 1/2 that don't match, and thus have CIoU = 0.
    """
    psg = pixel_sharing_bipartite(lab_gt, lab)
    iou = intersection_over_union(psg)
    matching = matching_overlap(psg, fractions=(0.5, 0))
    matching[0,:] = False
    matching[:,0] = False
    n_gt = len(set(np.unique(lab_gt)) - {0})
    n_matched = iou[matching].sum()
    if partial_dataset:
        return n_matched , n_gt
    else:
        return n_matched / n_gt