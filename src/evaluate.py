import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from time import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras import backend as K
import tensorflow as tf

random_state = 90
seed = np.random.seed(random_state)
tf.set_random_seed(random_state)


def evaluate():
    def tp(true, pred):
        y_true = true[:, 1:]
        y_pred = pred[:, 1:]
        tp_3d = K.concatenate(
            [
                K.cast(y_true, 'bool'),
                K.cast(K.round(y_pred), 'bool'),
                K.cast(K.ones_like(y_pred), 'bool')
            ], axis=1
        )
        return K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))

    def tn(true, pred):
        y_true = true[:, 1:]
        y_pred = pred[:, 1:]
        count = K.sum(K.cast(K.ones_like(y_pred), 'int32'))
        tn_3d = K.concatenate(
            [
                K.cast(y_true, 'bool'),
                K.cast(K.round(y_pred), 'bool'),
                K.cast(K.zeros_like(y_pred), 'bool')
            ], axis=1
        )
        return count - K.sum(K.cast(K.any(tn_3d, axis=1), 'int32'))

    def fp(true, pred):
        y_true = true[:, 1:]
        y_pred = pred[:, 1:]
        fp_3d = K.concatenate(
            [
                K.cast(K.abs(y_true - K.ones_like(y_true)), 'bool'),
                K.cast(K.round(y_pred), 'bool'),
                K.cast(K.ones_like(y_pred), 'bool')
            ], axis=1
        )
        return K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))

    def fn(true, pred):
        y_true = true[:, 1:]
        y_pred = pred[:, 1:]
        fn_3d = K.concatenate(
            [
                K.cast(y_true, 'bool'),
                K.cast(K.abs(K.round(y_pred) - K.ones_like(y_pred)), 'bool'),
                K.cast(K.ones_like(y_pred), 'bool')
            ], axis=1
        )
        return K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))

    unsw_nb15_4 = "dataset/UNSW-NB15_4.csv"
    unsw_nb15_tr = "dataset/UNSW_NB15_testing-set.csv"
    
    cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
            'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
            'smean', 'dmean', 'trans_depth', 'response_body_len', 'sjit', 'djit', 'stime', 'ltime', 'sinpkt',
            'dinpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
            'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
            'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']

    df4 = pd.read_csv(unsw_nb15_4, header=None, names=cols)
    train_csv = pd.read_csv(unsw_nb15_tr)
    dataset = df4.head(100000)
    X_train = train_csv.drop(['id', 'rate', 'attack_cat', 'label'], axis=1)
    X_test = dataset.drop(['srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime', 'attack_cat', 'label'], axis=1)
    y_test = dataset['label']
    
    tr_num_keys = X_train.iloc[:, :-2].select_dtypes(exclude=['object']).keys()
    
    num_keys = X_test.iloc[:, :-2].select_dtypes(exclude=['object']).keys()
    cat_keys = X_test.iloc[:, :-2].select_dtypes(include=['object']).keys()

    X_train[tr_num_keys] = X_train[tr_num_keys].apply(lambda x: np.log(x + 1))
    X_test[num_keys] = X_test[num_keys].apply(lambda x: np.log(x + 1))

    scaler = MinMaxScaler()
    X_train[tr_num_keys] = scaler.fit_transform(X_train[tr_num_keys])
    X_test[num_keys] = scaler.fit_transform(X_test[num_keys].fillna(0))

    for key in cat_keys:
        X_train[key] = pd.Series(data=pd.factorize(X_train[key])[0])
        X_test[key] = pd.Series(data=pd.factorize(X_test[key])[0])
    scaler = MinMaxScaler()
    X_train[cat_keys] = scaler.fit_transform(X_train[cat_keys])
    X_test[cat_keys] = scaler.fit_transform(X_test[cat_keys])

    pca = PCA(n_components=16, random_state=random_state)
    pca.fit(X_train)
    reduced_X_train = pca.transform(X_train)
    reduced_X_test = pca.transform(X_test)
    X_test = reduced_X_test

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(16,)))
    model.add(Dropout(.6))
    model.add(Dense(384, activation='relu'))
    model.add(Dropout(.6))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.6))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.6))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.6))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(.6))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(.6))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adagrad',
                  metrics=['accuracy', tp, tn, fp, fn])
    model.load_weights('results/without_rate.hdf5')

    test = pd.get_dummies(y_test)
    loss, acc, tp, tn, fp, fn = model.evaluate(X_test, test)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    far = (fpr + fnr) / 2
    print ("acc = {}, far = {}".format(acc,far))
    

