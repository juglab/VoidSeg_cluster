import numpy as np
from skimage.segmentation import find_boundaries
from sklearn.feature_extraction import image

def augment_data(X_train, Y_train):
    
    print('augmenting training data')
    X_ = X_train.copy()

    X_train_aug = np.concatenate((X_train, np.rot90(X_, 1, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))

    Y_ = Y_train.copy()

    Y_train_aug = np.concatenate((Y_train, np.rot90(Y_, 1, (1, 2))))
    Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 2, (1, 2))))
    Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 3, (1, 2))))
    Y_train_aug = np.concatenate((Y_train_aug, np.flip(Y_train_aug, axis=1)))


    print('Training data size after augmentation', X_train_aug.shape)
    print('Training data size after augmentation', Y_train_aug.shape)

    return X_train_aug, Y_train_aug

def normalize(img, mean, std):
    return (img - mean)/std

def denormalize(img, mean, std):
    return (img * std) + mean

def convert_to_oneHot(data):
    data_oneHot = np.zeros((*data.shape, 3), dtype=np.float32)
    for i in range(data.shape[0]):
        data_oneHot[i] = onehot_encoding(add_boundary_label(data[i].astype(np.int32)))
    return data_oneHot

def add_boundary_label(lbl, dtype=np.uint16):
    """ lbl is an integer label image (not binarized) """
    b = find_boundaries(lbl, mode='outer')
    res = (lbl > 0).astype(dtype)
    res[b] = 2
    return res

def onehot_encoding(lbl, n_classes=3, dtype=np.uint32):
    """ n_classes will be determined by max lbl value if its value is None """
    onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
    for i in range(n_classes):
        onehot[lbl == i, ..., i] = 1
    return onehot

def create_patches(images, masks, size):
    patchesimages = image.extract_patches_2d(images, (size, size), 10, 0)
    patchesmasks = image.extract_patches_2d(masks, (size, size), 10, 0)

    return patchesimages, patchesmasks
