import numpy as np

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