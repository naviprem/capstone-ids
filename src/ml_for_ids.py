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
np.random.seed(random_state)
tf.set_random_seed(random_state)

"""
ModalData class encapsulates all exploration, preprocessing and training steps of the data
"""

class ModalData:

    """
    Class initialization function: loads the dataset
    """
    def __init__(self):
        self.unsw_nb15_tr = unsw_nb15_tr
        self.unsw_nb15_ts = unsw_nb15_ts
        self.rs = random_state
        self.load()

    """
    Load function: extracts features and labels and creates X and y dataframes.
    """
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

    """
    Data Exploratory Function: explores basic attributes of the dataset
    """
    def data_explore(self):
        print("Total number of records: ", len(self.X_train))
        attack = len(self.train_csv[self.train_csv['label'] == 1])
        normal = len(self.train_csv[self.train_csv['label'] == 0])
        print("Total number of normal records: ", normal)
        print("Total number of attack records: ", attack)
        print("Percentage of attack records: ", (float(attack)/float(len(self.X_train))) * 100)

    """
    Data Exploratory Function: explores numeric features
    """
    def explore_num_features(self):
        data_explore = self.X_train[self.num_keys].describe()
        with pd.option_context('display.max_rows', None, 'display.max_columns', 50, 'display.line_width', 500,
                               'display.precision', 1):
            print(data_explore)
        data_explore_pivot = data_explore.transpose()
        data_explore_pivot.to_csv("results/explore_num_features.csv")

    """
    Data Exploratory Function: explores categorical features
    """
    def explore_cat_features(self):
        data_explore = self.X_train[self.cat_keys].describe()
        with pd.option_context('display.max_rows', None, 'display.max_columns', 50, 'display.line_width', 500,
                               'display.precision', 1):
            print(data_explore)
            data_explore_pivot = data_explore.transpose()
            data_explore_pivot.to_csv("results/explore_cat_features.csv")

    """
    Data Visualization Function: generates and saves pairwise correlation matrix heatmap
    """
    def initial_heatmap(self):
        plt.figure(figsize=(13, 13))
        sns.heatmap(self.X_train[self.num_keys].corr())
        plt.savefig("results/initial_training_heatmap.png")
        plt.show()

    """
    Data Visualization Function: generates and saves feature wise histogram and distribution curves
    """
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

    """
    Data Preprocessing Function: performs log transformation on numerical features of training and test datasets
    """
    def log_transform(self):
        self.X_train[self.num_keys] = self.X_train[self.num_keys].apply(lambda x: np.log(x + 1))
        self.X_test[self.num_keys] = self.X_test[self.num_keys].apply(lambda x: np.log(x + 1))

    """
    Data Preprocessing Function: identifies outliers and prints them on console
    """
    def identify_outliers(self):
        numeric_Features = self.X_train[self.num_keys]
        all_outliers = np.array([])
        for feature in numeric_Features.keys():
            Q1 = numeric_Features[feature].quantile(.25)
            Q3 = numeric_Features[feature].quantile(.75)
            step = (float(Q3) - float(Q1)) * 1.5
            outliers = numeric_Features[
                ~((numeric_Features[feature] >= Q1 - step) & (numeric_Features[feature] <= Q3 + step))]
            all_outliers = np.append(all_outliers, np.array(outliers.index.values.tolist()).astype(int))
        self.outlier_count = np.count_nonzero(np.unique(all_outliers))
        print("Total number of outlier identified using Tukey's method = ", self.outlier_count)

    """
    Data Preprocessing Function: performs minmax transformation on numerical features of training and test datasets
    """
    def min_max_scaler(self):
        scaler = MinMaxScaler()
        self.X_train[self.num_keys] = scaler.fit_transform(self.X_train[self.num_keys])
        self.X_test[self.num_keys] = scaler.fit_transform(self.X_test[self.num_keys])

    """
    Data Preprocessing Function: performs factorize encoding on categorical features of training and test datasets
    """
    def factorize(self):
        for key in self.cat_keys:
            self.X_train[key] = pd.Series(data=pd.factorize(self.X_train[key])[0])
            self.X_test[key] = pd.Series(data=pd.factorize(self.X_test[key])[0])
        scaler = MinMaxScaler()
        self.X_train[self.cat_keys] = scaler.fit_transform(self.X_train[self.cat_keys])
        self.X_test[self.cat_keys] = scaler.fit_transform(self.X_test[self.cat_keys])

    """
    Data Preprocessing Function: performs principal component analysis and reduces the dimensionality of the datasets
    Saves explained variance to csv file
    """
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

    """
    Utility Function: counts and returns the number of columns in the dataset
    """
    def get_col_count(self):
        return len(self.X_train.columns)

    """
    Training Function: trains a random forest model on the training dataset
    Evaluates the model on the test dataset and records the results in metrics.csv
    """
    def train_random_forest(self):
        metrics = {}

        # Initialize the classifier
        clf = RandomForestClassifier(random_state=random_state)

        # Start training
        start = time()
        model = clf.fit(self.X_train, self.y_train)
        end = time()
        metrics["train_time"] = end - start

        # Predict labels for test dataset
        start = time()
        y_pred = clf.predict(self.X_test)
        end = time()

        # Collecting metrics
        metrics["predict_time"] = end - start
        metrics["model_name"] = model.__class__.__name__
        metrics["refined"] = False
        metrics["accuracy"] = float(accuracy_score(self.y_test, pd.DataFrame(data=y_pred)))
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        metrics["false_positive_rate"] = fpr = fp / (fp + tn)
        metrics["false_negative_rate"] = fnr = fn / (fn + tp)
        metrics["false_alarm_Rate"] = (fpr + fnr) / 2

        # Record metrics
        self.write_to_metrics_csv(metrics)

    """
    Training Function: trains a random forest model on the training dataset
    Evaluates the model on the test dataset and records the results in metrics.csv
    """
    def train_random_forest_refined(self):
        metrics = {}

        # Initialize the classifier
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=self.rs )
        model = RandomForestClassifier(random_state=random_state)
        scorer = make_scorer(accuracy_score)
        param_dist = {"max_features": ["sqrt", "log2"],
                      "min_samples_split": [2, 8, 14, 16],
                      "criterion": ["gini", "entropy"]}

        # Starts grid search/training
        start = time()
        search_obj = GridSearchCV(model, cv=5, param_grid=param_dist, scoring=scorer, return_train_score=True)
        fit = search_obj.fit(self.X_train, self.y_train)
        end = time()
        metrics["train_time"] = end - start


        # Predict labels for test dataset using best model from grid search
        self.best_model = fit.best_estimator_
        start = time()
        best_predictions = self.best_model.predict(self.X_test)
        end = time()

        # Collecting metrics
        metrics["model_name"] = model.__class__.__name__
        metrics["refined"] = False
        metrics["predict_time"] = end - start
        metrics["accuracy"] = float(accuracy_score(self.y_test, pd.DataFrame(data=best_predictions)))
        tn, fp, fn, tp = confusion_matrix(self.y_test, best_predictions).ravel()
        metrics["false_positive_rate"] = fpr = fp / (fp + tn)
        metrics["false_negative_rate"] = fnr = fn / (fn + tp)
        metrics["false_alarm_Rate"] = (fpr + fnr) / 2

        # Record metrics
        self.write_to_metrics_csv(metrics)

    """
    Keras Metrics Callback Function: calculates true positives
    """
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

    """
    Keras Metrics Callback Function: calculates true negatives
    """
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

    """
    Keras Metrics Callback Function: calculates false positives
    """
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

    """
    Keras Metrics Callback Function: calculates false negatives
    """
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

    """
    Training Function: trains a multilayer perceptron model on the training dataset
    Evaluates the model on the test dataset and records the results in metrics.csv
    """
    def train_mlp(self):
        metrics = {}

        # Initialize and compile the neural network
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(16,)))
        model.add(Dropout(.4))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adagrad',
                      metrics=['accuracy', self.tp, self.tn, self.fp, self.fn])
        model.summary()

        # Prepare encode labels
        y_train = pd.get_dummies(self.y_train)
        y_test = pd.get_dummies(self.y_test)

        # Train the dataset
        start = time()
        model.fit(self.X_train, y_train, epochs=5, batch_size=5000,
                  validation_split=0.1, verbose=2, shuffle=True)
        end = time()
        metrics["train_time"] = end - start

        # Evaluate Model
        start = time()
        loss, acc, tp, tn, fp, fn = model.evaluate(self.X_test, y_test)
        end = time()
        metrics["predict_time"] = end - start

        # Collect Metrics
        metrics["model_name"] = "Multilayer Perceptron"
        metrics["refined"] = False
        metrics["accuracy"] = acc
        metrics["false_positive_rate"] = fpr = fp / (fp + tn)
        metrics["false_negative_rate"] = fnr = fn / (fn + tp)
        metrics["false_alarm_Rate"] = (fpr + fnr) / 2

        # Record Metrics
        self.write_to_metrics_csv(metrics)

    """
    Training Function: trains a multilayer perceptron model on the training dataset
    Evaluates the model on the test dataset and records the results in metrics.csv
    """
    def train_mlp_refined(self):
        metrics = {}

        # Initialize and compile the neural network
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

        # Initialize checkpointer and earlystopping callback classes
        checkpointer = ModelCheckpoint(filepath='results/model.weights.best.hdf5', verbose=1,
                                       save_best_only=True, monitor='loss', mode='min')
        earlystopping = EarlyStopping(monitor='loss', patience=75, verbose=1, mode='min')

        # Shuffle and prepare dataset and labels
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=self.rs)
        y_train = pd.get_dummies(self.y_train)
        y_test = pd.get_dummies(self.y_test)

        # Train the dataset
        start = time()
        model.fit(self.X_train, y_train, epochs=750, batch_size=1000,
                       validation_split=0.35, verbose=2, callbacks=[checkpointer, earlystopping], shuffle=True)
        end = time()
        metrics["train_time"] = end - start

        # Evaluate the model
        start = time()
        model.load_weights('results/model.weights.best.hdf5')
        loss, acc, tp, tn, fp, fn = model.evaluate(self.X_test, y_test)
        end = time()
        metrics["predict_time"] = end - start

        # Collect metrics
        metrics["model_name"] = "Multilayer Perceptron"
        metrics["refined"] = True
        metrics["accuracy"] = acc
        metrics["false_positive_rate"] = fpr = fp / (fp + tn)
        metrics["false_negative_rate"] = fnr = fn / (fn + tp)
        metrics["false_alarm_Rate"] = (fpr + fnr) / 2

        # Record metrics
        self.write_to_metrics_csv(metrics)

    """
    Utility Function: records metrics to /results/metrics.csv file
    """
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

    """
    Result Visualization Function: generates and saves an attack catrgory wise confusion matrix
    """
    def confusion_matrix(self):
        # Initialize the best MLP model
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

        # Load the weights saves earlier from the hdf5 file
        model.load_weights('results/model.weights.best.hdf5')
        pred = model.predict(self.X_test)

        # Generate categories for the predictions
        y_pred = pred[:, 1:]
        y_pred = np.round(y_pred).astype(int),
        union = np.concatenate((np.reshape(np.array(self.y_cat), (82332,1)), np.reshape(np.array(y_pred), (82332,1))), axis=1)
        self.p_cat = np.array([self.get_attack_cat(x) for x in union])
        classes = np.sort(np.unique(self.y_cat))

        # Create confusion Matrix
        cm = confusion_matrix(np.array(self.y_cat), self.p_cat, labels=classes)

        # Visualize Confusion Matrix
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

    """
    Utility Function: To aid confusion matrix creation
    returns 'Normal' if the prediction is 0 (which indicates 'Normal')
    returns the associated category if the prediction is 1
    If the prediction is 1 and the associated true category is 'Normal'. It is a False Positive, return 'Generic'
    """
    def get_attack_cat(self, x):
        if x[1] == 0:
            return "Normal"
        elif x[0] == "Normal":
            return "Generic"
        else:
            return x[0]
