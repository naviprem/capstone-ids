import src.ml_for_ids as ml
import src.dl_for_ids as dl

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
    data.pca(2)
    # data.processed_heatmap()
    # data.processed_distplot()

def identify_outliers():
    data = ml.ModalData()
    data.log_transform()
    data.identify_outliers()