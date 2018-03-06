import src.ml_for_ids as ml
import src.evaluate as e

def data_exploration():
    data = ml.ModalData()
    data.explore_num_features()
    data.explore_cat_features()
    data.data_explore()

def data_visualization():
    data = ml.ModalData()
    data.initial_heatmap()
    data.initial_distplot()

def data_preprocessing():
    data = ml.ModalData()
    data.log_transform()
    data.min_max_scaler()
    data.factorize()
    data.pca(41)

def identify_outliers():
    data = ml.ModalData()
    data.log_transform()
    data.identify_outliers()

def train_random_forest_model():
    data = ml.ModalData()
    data.log_transform()
    data.min_max_scaler()
    data.factorize()
    data.pca(16)
    data.train_random_forest()

def train_random_forest_model_refined():
    data = ml.ModalData()
    data.log_transform()
    data.min_max_scaler()
    data.factorize()
    data.pca(16)
    data.train_random_forest_refined()

def train_mlp_model():
    data = ml.ModalData()
    data.log_transform()
    data.min_max_scaler()
    data.factorize()
    data.pca(16)
    data.train_mlp()

def train_mlp_model_refined():
    data = ml.ModalData()
    data.log_transform()
    data.min_max_scaler()
    data.factorize()
    data.pca(16)
    data.train_mlp_refined()

def confusion_matrix():
    data = ml.ModalData()
    data.log_transform()
    data.min_max_scaler()
    data.factorize()
    data.pca(16)
    data.confusion_matrix()

def predict_from_raw_dataset():
    e.evaluate()

