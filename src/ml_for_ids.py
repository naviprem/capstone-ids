import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import os.path
import csv
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, Callback
from keras import backend as K
import tensorflow as tf
import itertools



# Interchanging training and testing datasets to Address incorrect filenames on dataset download url -
# https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys?path=%2FUNSWNB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set
unsw_nb15_tr = "dataset/UNSW_NB15_testing-set.csv"
unsw_nb15_ts = "dataset/UNSW_NB15_training-set.csv"
random_state = 90
#39, 60, 55, 90
seed = np.random.seed(random_state)
tf.set_random_seed(random_state)



class ModalData:

    def __init__(self):
        self.unsw_nb15_tr = unsw_nb15_tr
        self.unsw_nb15_ts = unsw_nb15_ts
        self.rs = random_state
        self.load()

    def load(self):
        self.train_csv = pd.read_csv(self.unsw_nb15_tr)
        self.test_csv = pd.read_csv(self.unsw_nb15_ts)
        self.X_train = self.train_csv.drop(['id', 'attack_cat', 'label'], axis=1)
        self.X_test = self.test_csv.drop(['id', 'attack_cat', 'label'], axis=1)
        self.y_train = self.train_csv['label']
        self.y_test = self.test_csv['label']
        self.y_cat = self.train_csv['attack_cat']
        self.y_cat = self.test_csv['attack_cat']
        self.all_keys = self.X_train.keys()
        self.num_keys = self.X_train.iloc[:, :-2].select_dtypes(exclude=['object']).keys()
        self.cat_keys = self.X_train.iloc[:, :-2].select_dtypes(include=['object']).keys()

    def explore_num_features(self):
        data_explore = self.X_train[self.num_keys].describe()
        with pd.option_context('display.max_rows', None, 'display.max_columns', 50, 'display.line_width', 500,
                               'display.precision', 1):
            print(data_explore)
        data_explore_pivot = data_explore.transpose()
        data_explore_pivot.to_csv("results/explore_num_features.csv")

    def data_explore(self):
        print("Total number of records: ", len(self.X_train))
        attack = len(self.train_csv[self.train_csv['label'] == 1])
        normal = len(self.train_csv[self.train_csv['label'] == 0])
        print("Total number of normal records: ", normal)
        print("Total number of attack records: ", attack)
        print("Percentage of attack records: ", (float(attack)/float(len(self.X_train))) * 100)

    def explore_cat_features(self):
        data_explore = self.X_train[self.cat_keys].describe()
        with pd.option_context('display.max_rows', None, 'display.max_columns', 50, 'display.line_width', 500,
                               'display.precision', 1):
            print(data_explore)
            data_explore_pivot = data_explore.transpose()
            data_explore_pivot.to_csv("results/explore_cat_features.csv")

    def log_transform(self):
        self.X_train[self.num_keys] = self.X_train[self.num_keys].apply(lambda x: np.log(x + 1))
        self.X_test[self.num_keys] = self.X_test[self.num_keys].apply(lambda x: np.log(x + 1))

    def initial_heatmap(self):
        plt.figure(figsize=(13, 13))
        sns.heatmap(self.X_train[self.num_keys].corr())
        plt.savefig("results/initial_training_heatmap.png")
        plt.show()

    def initial_distplot(self):
        sns.set(style="white", palette="muted", color_codes=True)
        rs = np.random.RandomState(10)

        # Set up the matplotlib figure
        f, axes = plt.subplots(10, 4, figsize=(12, 30))
        sns.despine(left=True)

        i=0
        for feature in self.num_keys:
            d = self.X_train[feature].values
            sns.distplot(d, color="m", ax=axes[int(i/4), i%4], axlabel=feature)
            i += 1

        plt.setp(axes, yticks=[], xticks=[])
        plt.tight_layout()
        plt.savefig("results/histogram_distribution.png")
        plt.show()

    def processed_heatmap(self):
        plt.figure(figsize=(13, 13))
        sns.heatmap(pd.DataFrame(data=self.X_train).corr())
        plt.savefig("results/processed_training_heatmap.png")
        plt.show()

    def processed_distplot(self):
        sns.set(style="white", palette="muted", color_codes=True)
        rs = np.random.RandomState(10)

        # Set up the matplotlib figure
        f, axes = plt.subplots(3, 4, figsize=(12, 9))
        sns.despine(left=True)

        for i in range(0,11):
            d = self.X_train[i]
            sns.distplot(d, color="m", ax=axes[int(i / 4), i % 4])


        plt.setp(axes, yticks=[], xticks=[])
        plt.tight_layout()
        plt.savefig("results/processed_distribution.png")
        plt.show()

    def identify_outliers(self):
        # select_features = ["response_body_len", "dur"]
        numeric_Features = self.X_train[self.num_keys]
        all_outliers = np.array([])
        for feature in numeric_Features.keys():
            Q1 = numeric_Features[feature].quantile(.25)
            Q3 = numeric_Features[feature].quantile(.75)
            step = (float(Q3) - float(Q1)) * 1.5

            # print(
            #     "Data points considered outliers for the feature "
            #     "'{}':\nQ1 = {}, Q3 = {}, Step = {}".format(feature, Q1, Q3, step))
            outliers = numeric_Features[
                ~((numeric_Features[feature] >= Q1 - step) & (numeric_Features[feature] <= Q3 + step))]
            all_outliers = np.append(all_outliers, np.array(outliers.index.values.tolist()).astype(int))
            # print(numeric_Features[
            #           ~((numeric_Features[feature] >= Q1 - step) & (numeric_Features[feature] <= Q3 + step))][feature])
        self.outlier_count = np.count_nonzero(np.unique(all_outliers))
        print("Total number of outlier identified using Tukey's method = ", self.outlier_count)

    def min_max_scaler(self):
        scaler = MinMaxScaler()
        # print(ds["X_train"].isnull().any())
        self.X_train[self.num_keys] = scaler.fit_transform(self.X_train[self.num_keys])
        self.X_test[self.num_keys] = scaler.fit_transform(self.X_test[self.num_keys])

    def factorize(self):
        for key in self.cat_keys:
            self.X_train[key] = pd.Series(data=pd.factorize(self.X_train[key])[0])
            self.X_test[key] = pd.Series(data=pd.factorize(self.X_test[key])[0])
        scaler = MinMaxScaler()
        self.X_train[self.cat_keys] = scaler.fit_transform(self.X_train[self.cat_keys])
        self.X_test[self.cat_keys] = scaler.fit_transform(self.X_test[self.cat_keys])

    def pca(self, c):
        pca = PCA(n_components=c, random_state=self.rs)
        pca.fit(self.X_train)
        reduced_X_train = pca.transform(self.X_train)
        self.X_train = reduced_X_train

        reduced_X_test = pca.transform(self.X_test)
        self.X_test = reduced_X_test

        dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_) + 1)]
        components = pd.DataFrame(np.round(pca.components_, 4), columns=self.all_keys)
        components.index = dimensions
        ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
        variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
        variance_ratios.index = dimensions
        variance = pd.concat([variance_ratios, components], axis=1)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', 50, 'display.line_width', 500,
        #                        'display.precision', 4):
        #     print(variance)
        variance.to_csv("results/explained_variance_" + str(c) + ".csv")

    def get_col_count(self):
        return len(self.X_train.columns)

    def train_random_forest(self):
        metrics = {}
        clf = RandomForestClassifier(random_state=random_state)
        start = time()
        model = clf.fit(self.X_train, self.y_train)
        end = time()
        metrics["train_time"] = end - start
        metrics["model_name"] = model.__class__.__name__
        metrics["refined"] = False
        start = time()
        y_pred = clf.predict(self.X_test)
        end = time()
        metrics["predict_time"] = end - start

        metrics["accuracy"] = float(accuracy_score(self.y_test, pd.DataFrame(data=y_pred)))
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()

        metrics["false_positive_rate"] = fpr = fp / (fp + tn)
        metrics["false_negative_rate"] = fnr = fn / (fn + tp)
        metrics["false_alarm_Rate"] = (fpr + fnr) / 2
        self.write_to_metrics_csv(metrics)

    def tp(self, true, pred):
        y_true = true[:,1:]
        y_pred = pred[:,1:]
        tp_3d = K.concatenate(
            [
                K.cast(y_true, 'bool'),
                K.cast(K.round(y_pred), 'bool'),
                K.cast(K.ones_like(y_pred), 'bool')
            ], axis=1
        )
        return K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))

    def tn(self, true, pred):
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

    def fp(self, true, pred):
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

    def fn(self, true, pred):
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

    def train_mlp_refined(self):
        metrics = {}

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
                      metrics=['accuracy', self.tp, self.tn, self.fp, self.fn])
        model.summary()
        checkpointer = ModelCheckpoint(filepath='results/model.weights.best.hdf5', verbose=1,
                                       save_best_only=True, monitor='loss', mode='min')
        earlystopping = EarlyStopping(monitor='loss', patience=75, verbose=1, mode='min')
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=self.rs)
        y_train = pd.get_dummies(self.y_train)
        y_test = pd.get_dummies(self.y_test)
        start = time()
        model.fit(self.X_train, y_train, epochs=750, batch_size=1000,
                       validation_split=0.35, verbose=2, callbacks=[checkpointer, earlystopping], shuffle=True)
        end = time()
        metrics["train_time"] = end - start
        metrics["model_name"] = "Multilayer Perceptron"
        metrics["refined"] = True

        start = time()
        model.load_weights('results/model.weights.best.hdf5')
        loss, acc, tp, tn, fp, fn = model.evaluate(self.X_test, y_test)
        end = time()

        metrics["predict_time"] = end - start
        metrics["accuracy"] = acc
        metrics["false_positive_rate"] = fpr = fp / (fp + tn)
        metrics["false_negative_rate"] = fnr = fn / (fn + tp)
        metrics["false_alarm_Rate"] = (fpr + fnr) / 2
        self.write_to_metrics_csv(metrics)

    def train_mlp(self):
        metrics = {}

        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(16,)))
        model.add(Dropout(.4))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adagrad',
                      metrics=['accuracy', self.tp, self.tn, self.fp, self.fn])
        model.summary()
        y_train = pd.get_dummies(self.y_train)
        y_test = pd.get_dummies(self.y_test)
        start = time()
        model.fit(self.X_train, y_train, epochs=5, batch_size=5000,
                  validation_split=0.1, verbose=2, shuffle=True)
        end = time()
        metrics["train_time"] = end - start
        metrics["model_name"] = "Multilayer Perceptron"
        metrics["refined"] = False

        start = time()
        loss, acc, tp, tn, fp, fn = model.evaluate(self.X_test, y_test)
        end = time()

        metrics["predict_time"] = end - start
        metrics["accuracy"] = acc
        metrics["false_positive_rate"] = fpr = fp / (fp + tn)
        metrics["false_negative_rate"] = fnr = fn / (fn + tp)
        metrics["false_alarm_Rate"] = (fpr + fnr) / 2
        self.write_to_metrics_csv(metrics)

    def write_to_metrics_csv(self, results):
        metrics_csv = "results/metrics.csv"
        log_keys = ["model_name", "refined", "train_time", "predict_time", "accuracy",
                    'false_positive_rate', "false_negative_rate", "false_alarm_Rate"]

        if os.path.exists(metrics_csv):
            with open(metrics_csv, 'a') as log_file:
                writer = csv.DictWriter(log_file, log_keys, extrasaction="ignore")
                writer.writerow(results)
        else:
            with open(metrics_csv, 'a') as log_file:
                wr = csv.writer(log_file)
                wr.writerow(log_keys)
            with open(metrics_csv, 'a') as log_file:
                writer = csv.DictWriter(log_file, log_keys, extrasaction="ignore")
                writer.writerow(results)

    def train_random_forest_refined(self):
        metrics = {}
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=self.rs )
        model = RandomForestClassifier(random_state=random_state)
        scorer = make_scorer(accuracy_score)
        param_dist = {"max_features": ["sqrt", "log2"],
                      "min_samples_split": [2, 8, 14, 16],
                      "criterion": ["gini", "entropy"]}

        start = time()
        search_obj = GridSearchCV(model, cv=5, param_grid=param_dist, scoring=scorer, return_train_score=True)
        fit = search_obj.fit(self.X_train, self.y_train)
        end = time()

        metrics["train_time"] = end - start
        metrics["model_name"] = model.__class__.__name__
        metrics["refined"] = False

        self.cv_results = search_obj.cv_results_
        self.best_model = fit.best_estimator_
        self.important_features = self.best_model.feature_importances_
        self.best_params = search_obj.best_params_

        start = time()
        best_predictions = self.best_model.predict(self.X_test)
        end = time()

        metrics["predict_time"] = end - start

        metrics["accuracy"] = float(accuracy_score(self.y_test, pd.DataFrame(data=best_predictions)))
        tn, fp, fn, tp = confusion_matrix(self.y_test, best_predictions).ravel()

        metrics["false_positive_rate"] = fpr = fp / (fp + tn)
        metrics["false_negative_rate"] = fnr = fn / (fn + tp)
        metrics["false_alarm_Rate"] = (fpr + fnr) / 2
        self.write_to_metrics_csv(metrics)

    def confusion_matrix(self):
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
                      metrics=['accuracy'])
        model.load_weights('results/model.weights.best.hdf5')
        pred = model.predict(self.X_test)

        y_pred = pred[:, 1:]
        y_pred = np.round(y_pred).astype(int),
        union = np.concatenate((np.reshape(np.array(self.y_cat), (82332,1)), np.reshape(np.array(y_pred), (82332,1))), axis=1)
        self.p_cat = np.array([self.get_attack_cat(x) for x in union])
        classes = np.sort(np.unique(self.y_cat))
        cm = confusion_matrix(np.array(self.y_cat), self.p_cat, labels=classes)


        plt.figure(figsize=(8, 7))
        title = 'Confusion matrix'
        cmap = plt.cm.Blues
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("results/confusion_matrix.png")
        plt.show()

    def get_attack_cat(self, x):
        if x[1] == 0:
            return "Normal"
        elif x[0] == "Normal":
            return "Generic"
        else:
            return x[0]
