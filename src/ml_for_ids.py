import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import randint as sp_randint
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans



# Interchanging training and testing datasets to Address incorrect filenames on dataset download url -
# https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys?path=%2FUNSWNB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set
unsw_nb15_tr = "dataset/UNSW_NB15_testing-set.csv"
unsw_nb15_ts = "dataset/UNSW_NB15_training-set.csv"
random_state = 55
#39, 60, 55, 90
seed = np.random.seed(random_state)

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
        # sns.heatmap(self.X_test[self.num_keys].corr())
        # plt.savefig("results/initial_testing_heatmap.png")
        # plt.show()

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
        with pd.option_context('display.max_rows', None, 'display.max_columns', 50, 'display.line_width', 500,
                               'display.precision', 4):
            print(variance)
        variance.to_csv("results/explained_variance.csv")

        fig1, ax1 = plt.subplots(figsize=(14, 8))
        self.pca_data = pd.DataFrame(np.round(reduced_X_train, 4), columns=variance.index.values)
        pca_label_data = pd.concat([self.pca_data, self.y_train], axis=1)
        pca_attack_data = pca_label_data[pca_label_data['label'] == 1]
        pca_normal_data = pca_label_data[pca_label_data['label'] == 0]
        ax1.scatter(x=pca_attack_data.loc[:, 'Dimension 1'], y=pca_attack_data.loc[:, 'Dimension 2'],
               facecolors='r', edgecolors='r', s=10, alpha=0.5)
        ax1.scatter(x=pca_normal_data.loc[:, 'Dimension 1'], y=pca_normal_data.loc[:, 'Dimension 2'],
                    facecolors='b', edgecolors='b', s=10, alpha=0.5)
        plt.savefig("results/pca-with-labels.png")
        plt.show()

        clusterer = KMeans(n_clusters=2, random_state=0).fit(self.X_test)


        preds = clusterer.predict(self.X_test)


        centers = clusterer.cluster_centers_
        print(centers)


        # sample_preds = clusterer.predict(pca_samples)


        from sklearn import metrics
        # score = metrics.silhouette_score(self.X_test, preds)
        # display(score)
        # Display the results of the clustering from implementation
        # vs.cluster_results(reduced_data, preds, centers, pca_samples)
        predictions = pd.DataFrame(preds, columns=['Cluster'])
        plot_data = pd.concat([predictions, pd.DataFrame(self.X_test, columns=variance.index.values)], axis=1)

        # Generate the cluster plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Color map
        cmap = cm.get_cmap('gist_rainbow')

        # Color the points based on assigned cluster
        for i, cluster in plot_data.groupby('Cluster'):
            cluster.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2', color=cmap((i) * 1.0 / (len(centers) - 1)), label='Cluster %i' % (i), s=10);

        # Plot centers with indicators
        for i, c in enumerate(centers):
            ax.scatter(x=c[0], y=c[1], color='white', edgecolors='black', alpha=1, linewidth=2, marker='o', s=200);
            ax.scatter(x=c[0], y=c[1], marker='$%d$' % (i), alpha=1, s=100);

        # Plot transformed sample points
        # ax.scatter(x=pca_samples[:, 0], y=pca_samples[:, 1], s=150, linewidth=4, color='black', marker='x');

        # Set plot title
        ax.set_title(
            "Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");
        plt.savefig("results/cluster.png")
        plt.show()

    def get_col_count(self):
        return len(self.X_train.columns)

    def train(self):
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=self.rs )
        clf = AdaBoostClassifier(random_state=self.rs)
        scorer = make_scorer(accuracy_score)
        param_dist = {"max_depth": [3, None],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(2, 11),
                      "min_samples_leaf": sp_randint(1, 11),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}
        parameters = {'max_features': ['log2'], 'criterion': ["entropy"]}
        param_dist = {"n_estimators": sp_randint(1, 100),
                      "random_state": sp_randint(0, 100)}

        start = time()  # Get start time
        # search_obj = GridSearchCV(clf, param_grid=parameters, scoring=scorer, return_train_score=True)
        search_obj = RandomizedSearchCV(clf, param_distributions=param_dist, scoring=scorer, n_iter=20,
                                        random_state=self.rs, return_train_score=True)
        fit = search_obj.fit(self.X_train, self.y_train)
        end = time()  # Get end time

        self.train_time = end - start
        self.cv_results = search_obj.cv_results_
        self.best_clf = fit.best_estimator_
        self.important_features = self.best_clf.feature_importances_
        self.best_params = search_obj.best_params_

        start = time()  # Get start time
        best_predictions = self.best_clf.predict(self.X_test)
        self.accuracy = float(accuracy_score(self.y_test, pd.DataFrame(data=best_predictions)))
        end = time()  # Get end time

        self.predict_time = end - start





# data = ModalData()
# data.explore_num_features()
# # data.explore_cat_features()
# # data.display_heatmap()
# # data.log_transform()
# # # data.display_heatmap()
# # # data.identify_outliers()
# # data.min_max_scaler()
# # data.factorize()
# # data.pca()
# # data.train()
# print("Accuracy: " + str(data.accuracy))
# # print("Cross Validation Results: ")
# # with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
# #     print(pd.DataFrame.from_dict(data.cv_results))
# # print("Important Features: ", *data.important_features)
# print("Training time: " + str(data.train_time))
# print("Predict time: " + str(data.predict_time))
# print("Best Params: ", data.best_params)
