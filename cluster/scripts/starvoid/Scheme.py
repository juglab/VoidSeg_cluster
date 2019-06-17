import json
import numpy as np
from train_utils import  *
import pickle
from scipy import ndimage
from skimage import io
from os.path import join

class Scheme():
    def __init__(self):
        with open('experiment.json', 'r') as f:
            self.exp_conf = json.load(f)

        self.scheme = self.exp_conf['scheme']

    def shuffle_train_data(self, X_train, Y_train):
        if 'is_seeding' in self.exp_conf.keys():
            if self.exp_conf['is_seeding']:
                print('seeding training data')
                np.random.seed(self.exp_conf['random_seed'])
                seed_ind = np.random.permutation(X_train.shape[0])
                X_train = X_train[seed_ind]
                Y_train = Y_train[seed_ind]

        return X_train, Y_train

    def fractionate_train_data(self, X_train, Y_train):
        train_frac = int(np.round((self.exp_conf['train_frac'] / 100) * X_train.shape[0]))
        X_train = X_train[:train_frac]
        Y_train = Y_train[:train_frac]

        return X_train, Y_train

    def prepare_val_data(self, X_val, Y_val, Y_val_oneHot):

        X_validation = X_val[..., np.newaxis]
        Y_validation = Y_val[..., np.newaxis]
        Y_validation = np.concatenate((X_validation.copy(), np.ones(X_validation.shape, dtype=np.float32)), axis=3)
        Y_validation = np.concatenate((Y_validation, Y_val_oneHot), axis=3)

        return X_validation, Y_validation

    def preprocess(self, model, train_data, val_data):
        assert False, 'Implementation Missing.'

    def train_seg(self, model, train_data, val_data):
        assert False, 'Implementation Missing.'

    def predict_seg(self, model, test, mean_std):
        mean, std = mean_std[0], mean_std[1]
        X_test = test['X_test']
        X = normalize(X_test, mean, std)
        with open('best_score.dat', 'rb') as best_score_file:
            ts = pickle.load(best_score_file)[0]
        print('Use threshold =', ts)
        for i in range(X.shape[0]):
            prediction = model.predict(X[i], axes='YX', normalizer=None)
            prediction_exp = np.exp(prediction[..., 1:])
            prediction_seg = prediction_exp / np.sum(prediction_exp, axis=2)[..., np.newaxis]
            prediction_bg = prediction_seg[..., 0]
            prediction_fg = prediction_seg[..., 1]
            prediction_b = prediction_seg[..., 2]
            pred_thresholded = prediction_fg > ts
            labels, nb = ndimage.label(pred_thresholded)
            #    predictions.append(pred)
            io.imsave(join(self.exp_conf['base_dir'], 'mask' + str(i).zfill(3) + '.tif'), labels.astype(np.int16))
            io.imsave(join(self.exp_conf['base_dir'], 'foreground' + str(i).zfill(3) + '.tif'), prediction_fg)
            io.imsave(join(self.exp_conf['base_dir'], 'background' + str(i).zfill(3) + '.tif'), prediction_bg)
            io.imsave(join(self.exp_conf['base_dir'], 'border' + str(i).zfill(3) + '.tif'), prediction_b)

    def create_n2v_train_data(self, train_data):
        X_train = train_data['X_train']
        Y_train = train_data['Y_train']
        X_val = train_data['X_val']
        Y_val = train_data['Y_val']

        mean, std = np.mean(X_train), np.std(X_train)
        X_train = normalize(X_train, mean, std)
        X_val = normalize(X_val, mean, std)
        X_train, Y_train = self.shuffle_train_data(X_train, Y_train)
        if (self.exp_conf['augment']):
            X_train, Y_train = augment_data(X_train, Y_train)
            X_val, Y_val = augment_data(X_val, Y_val)

        shape = X_train.shape
        shape[-1] = 3

        Y_train = np.concatenate(
            (X_train[..., np.newaxis], np.zeros(shape, dtype=np.float32)[..., np.newaxis]),
            axis=3)

        shape = X_val.shape
        shape[-1] = 3
        X_val, Y_val = self.prepare_val_data(X_val, Y_val, np.zeros(shape, dtype=np.float32))

        return (X_train[..., np.newaxis], Y_train), (X_val, Y_val)

    def load_n2v_train_data(self):
        assert False, 'Implementation Missing.'
        return None

    def load_seg_train_data(self):
        assert False, 'Implementation Missing.'
        return None

    def load_seg_test_data(self):
        assert False, 'Implementation Missing.'
        return None

    def create_seg_train_data(self, train_data):
        X_train = train_data['X_train']
        Y_train = train_data['Y_train']
        X_val = train_data['X_val']
        Y_val = train_data['Y_val']

        mean, std = np.mean(X_train), np.std(X_train)
        X_train = normalize(X_train, mean, std)
        X_val = normalize(X_val, mean, std)
        X_train, Y_train = self.shuffle_train_data(X_train, Y_train)
        X_train, Y_train = self.fractionate_train_data(X_train, Y_train)
        if (self.exp_conf['augment']):
            X_train, Y_train = augment_data(X_train, Y_train)
            X_val, Y_val = augment_data(X_val, Y_val)

        Y_train_oneHot = convert_to_oneHot(Y_train)
        Y_val_oneHot = convert_to_oneHot(Y_val)

        Y_train = np.concatenate(
            (X_train[..., np.newaxis], np.zeros(X_train.shape, dtype=np.float32)[..., np.newaxis], Y_train_oneHot),
            axis=3)

        X_val, Y_val = self.prepare_val_data(X_val, Y_val, Y_val_oneHot)

        return (X_train[..., np.newaxis], Y_train), (X_val, Y_val), (mean, std)

    def load_n2v_model(self):
        assert False, 'Implementation Missing.'

    def load_seg_model(self):
        assert False, 'Implementation Missing.'

    def compute(self):
        n2v_train_data = self.load_n2v_train_data()
        n2v_train, n2v_val = self.create_n2v_train_data(n2v_train_data)
        n2v_model = self.load_n2v_model()
        self.preprocess(n2v_model, n2v_train, n2v_val)

        seg_train_data = self.load_seg_train_data()
        seg_train, seg_val, mean_std = self.create_seg_train_data(seg_train_data)
        seg_model = self.load_seg_model()
        seg_test = self.load_seg_test_data()
        self.train_seg(seg_model, seg_train, seg_val)
        import compute_seg_threshold
        self.predict_seg(seg_model, seg_test, mean_std)