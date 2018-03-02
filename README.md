# capstone-ids

### Setting up the project

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/naviprem/capstone-ids.git
cd capstone-ids
```


2. Download the UNSW-NB15 [training and testing datasets](https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys?path=%2FUNSW-NB15%20-%20CSV%20Files%2Fa%20part%20of%20training%20and%20testing%20set).  
Place them in the repo, at location `path/to/ids-capstone/dataset`. 

3. To install TensorFlow on your local machine, follow [the guide](https://www.tensorflow.org/install/)   

4. Create (and activate) a new conda environment (on mac).
```
conda env create -f requirements/capston-ids.yml
source activate capstone-ids-project
``` 

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow (on mac).

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend"
```

### Executing various code sections

1. To train machine learning model on the UNSW-NB15 dataset

```bash
python src/ml-for-ids.py
```

2. To train deep learning model on the UNSW-NB15 dataset

```bash
python src/dl-for-ids.py
```

3. To explore feature set

```bash
python -c "from src.exec import *; data_exploration()"
```

3. For data visualization

```bash
python -c "from src.exec import *; data_visualization()"
```

4. For data preprocessing

```bash
python -c "from src.exec import *; data_preprocessing()"

```

5. To Identify Outliers

```bash
python -c "from src.exec import *; identify_outliers()"

```