# capstone-ids


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

6. To train machine learning model on the UNSW-NB15 dataset

```bash
python src/ml-for-ids.py
```
7. To train deep learning model on the UNSW-NB15 dataset
```bash
python src/dl-for-ids.py
```