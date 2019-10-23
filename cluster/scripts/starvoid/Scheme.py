import json
import numpy as np
from train_utils import  *
import pickle
from scipy import ndimage
from skimage import io
from os.path import join
from csbdeep.utils.n2v_utils import manipulate_val_data

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
        if (self.exp_conf['scheme'] == 'joint'):
            print("Joint loss! different fractionation")
            with open(self.exp_conf['base_dir'] + '/' + self.exp_conf['model_name'] + '_seg/' + 'config.json',
                      'r') as f:
                joint_conf = json.load(f)
            if (joint_conf['use_denoising']):
                print("zeroing out fractionated seg channels")
                Y_train[train_frac:, ..., 1:] *= 0
            else:
                X_train = X_train[:train_frac]
                Y_train = Y_train[:train_frac]
        else:
            X_train = X_train[:train_frac]
            Y_train = Y_train[:train_frac]

        return X_train, Y_train

    def prepare_seg_val_data(self, X_val, Y_val, Y_val_oneHot):

        X_validation = X_val[..., np.newaxis]
        Y_validation = Y_val[..., np.newaxis] 
        Y_validation = np.concatenate((X_validation.copy(), np.ones(X_validation.shape, dtype=np.float32)), axis=3)
        Y_validation = np.concatenate((Y_validation, Y_val_oneHot), axis=3)

        return X_validation, Y_validation
        
    def prepare_n2v_val_data(self, X_val, Y_val, Y_val_oneHot):

        X_validation = X_val[..., np.newaxis]
        Y_validation = np.concatenate((X_validation.copy(), np.zeros(X_validation.shape, dtype=np.float32)), axis=3)
        with open(self.exp_conf['base_dir']+'/'+self.exp_conf['model_name']+'_denoise/'+'config.json', 'r') as f:
            denoise_conf = json.load(f)
        num_pix = denoise_conf['n2v_num_pix']
        pixelsInPatch = denoise_conf['n2v_patch_shape'][0] * denoise_conf['n2v_patch_shape'][1]
        manipulate_val_data(X_validation, Y_validation, num_pix=int(num_pix * X_validation.shape[1] * X_validation.shape[2] / float(pixelsInPatch)),shape=(X_validation.shape[1], X_validation.shape[2]))
        Y_validation = np.concatenate((Y_validation, Y_val_oneHot), axis=3)
        return X_validation, Y_validation

    def preprocess_n2v(self, train_data, test_data):
        assert False, 'Implementation Missing.'
        return None, None

    def preprocess_seg(self, n2v_model, n2v_train_data, n2v_test_data, mean_std_denoise):
        assert False, 'Implementation Missing.'
        return None, None

    def train_denoise(self, model, train_data, val_data):
        assert False, 'Implementation Missing.'
        
    def train_seg(self, model, train_data, val_data):
        assert False, 'Implementation Missing.'
        
    def predict_denoise(self, model, train, test, mean_std):
        mean, std = mean_std[0], mean_std[1]
        X_train = train['X_train']
        X_val = train['X_val']
        X_test = test['X_test']
        X_train = normalize(X_train, mean, std)
        X_val = normalize(X_val, mean, std)
        X_test = normalize(X_test, mean, std)
        
        predictions = []
        for i in range(X_train.shape[0]):
            predictions.append(denormalize(model.predict(X_train[i], axes='YX',normalizer=None ), mean, std))
        X_train_d = np.array(predictions)
        
        predictions = []
        for i in range(X_val.shape[0]):
            predictions.append(denormalize(model.predict(X_val[i], axes='YX',normalizer=None ), mean, std))
        X_val_d = np.array(predictions)
        
        predictions = []
        for i in range(X_test.shape[0]):
            predictions.append(denormalize(model.predict(X_test[i], axes='YX',normalizer=None ), mean, std))
        X_test_d = np.array(predictions)
        
        return X_train_d, X_val_d, X_test_d

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

    def create_n2v_train_data(self, train_data_x, train_data_y, n2v_trainval_data):
        X_train = train_data_x
        X_val = n2v_trainval_data['X_val']
        Y_val = n2v_trainval_data['Y_val']

        mean, std = np.mean(X_train), np.std(X_train)
        X_train = normalize(X_train, mean, std)
        shape = list(X_train[..., np.newaxis].shape)
        shape[-1] = 4
        Y_train = np.concatenate((X_train[..., np.newaxis], np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=np.float32)), axis=3)
        
        X_val = normalize(X_val, mean, std)
        Y_val = normalize(Y_val, mean, std)
        X_train, Y_train = self.shuffle_train_data(X_train, Y_train)
        if (self.exp_conf['augment']):
            X_train, Y_train = augment_data(X_train, Y_train)
            X_val, Y_val = augment_data(X_val, Y_val)  

        shape = list(X_val[..., np.newaxis].shape)
        shape[-1] = 3
        X_val, Y_val = self.prepare_n2v_val_data(X_val, Y_val, np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=np.float32))

        return (X_train[..., np.newaxis], Y_train), (X_val, Y_val), (mean, std)

    def load_n2v_train_test_data(self):
        assert False, 'Implementation Missing.'
        return None, None

    def load_seg_train_test_data(self):
        assert False, 'Implementation Missing.'
        return None, None

    def create_seg_train_data(self, train_data):
        X_train = train_data['X_train']
        Y_train = train_data['Y_train']
        X_val = train_data['X_val']
        Y_val = train_data['Y_val']

        mean, std = np.mean(X_train), np.std(X_train)
        X_train = normalize(X_train, mean, std)
        X_val = normalize(X_val, mean, std)
        X_train, Y_train = self.shuffle_train_data(X_train, Y_train)
        if (self.exp_conf['scheme'] == 'joint'):
            print("Joint scheme!")
            print("Doing the right way!")
            train_frac = int(np.round((self.exp_conf['train_frac'] / 100) * X_train.shape[0]))
            Y_train1 = Y_train[:train_frac]
            X_train1 = X_train[:train_frac]
            Y_train2 = Y_train[train_frac:]
            X_train2 = X_train[train_frac:]
            Y_train2 *= 0


            if (self.exp_conf['augment']):
                X_train1, Y_train1 = augment_data(X_train1, Y_train1)
                X_train2, Y_train2 = augment_data(X_train2, Y_train2)
                X_train = np.concatenate((X_train1, X_train2), axis=0)
                Y_train = np.concatenate((Y_train1, Y_train2), axis=0)
                X_val, Y_val = augment_data(X_val, Y_val)

            Y_train_oneHot = convert_to_oneHot(Y_train)
            Y_val_oneHot = convert_to_oneHot(Y_val)

            Y_train = np.concatenate(
                (X_train[..., np.newaxis], np.zeros(X_train.shape, dtype=np.float32)[..., np.newaxis], Y_train_oneHot),
                axis=3)

            # X_train, Y_train = self.fractionate_train_data(X_train, Y_train)

            X_val, Y_val = self.prepare_seg_val_data(X_val, Y_val, Y_val_oneHot)

        else:
            print("Not joint scheme!")
            X_train, Y_train = self.fractionate_train_data(X_train, Y_train)
            if (self.exp_conf['augment']):
                X_train, Y_train = augment_data(X_train, Y_train)
                X_val, Y_val = augment_data(X_val, Y_val)

            Y_train_oneHot = convert_to_oneHot(Y_train)
            Y_val_oneHot = convert_to_oneHot(Y_val)

            Y_train = np.concatenate(
                (X_train[..., np.newaxis], np.zeros(X_train.shape, dtype=np.float32)[..., np.newaxis], Y_train_oneHot),
                axis=3)

            X_val, Y_val = self.prepare_seg_val_data(X_val, Y_val, Y_val_oneHot)


        return (X_train[..., np.newaxis], Y_train), (X_val, Y_val), (mean, std)

    def load_n2v_model(self):
        assert False, 'Implementation Missing.'

    def load_seg_model(self):
        assert False, 'Implementation Missing.'

    def compute(self):
        n2v_train_data, n2v_test_data = self.load_n2v_train_test_data()
        n2v_train_x, n2v_train_y = self.preprocess_n2v(n2v_train_data, n2v_test_data)
        n2v_train, n2v_val, mean_std_denoise = self.create_n2v_train_data(n2v_train_x, n2v_train_y, n2v_train_data)
        n2v_model = self.load_n2v_model()
        self.train_denoise(n2v_model, n2v_train, n2v_val)

        seg_train_data, seg_test_data = self.load_seg_train_test_data()
        seg_train_d, seg_test_d = self.preprocess_seg(n2v_model, seg_train_data, seg_test_data, mean_std_denoise)

        ####Comment all lines below to run N2V only through Sequential script.

        if (self.scheme != "sequential"):
            seg_train, seg_val, mean_std = self.create_seg_train_data(seg_train_d)
            seg_model = self.load_seg_model()
            self.train_seg(seg_model, seg_train, seg_val)

            import compute_seg_threshold
            # import compute_precision_threshold
            self.predict_seg(seg_model, seg_test_d, mean_std)

        else:
            print("Finisihing N2V part")
