# Word recognition from signals #

_Note: For this project I worked as an AI fellow at Insight Data Science, where I consulted with a private company in the San Francisco bay area. I have signed an NDA with them, so I am not allowed to give away any sensitive details. So I am keeping the codebase and the readme files completely generic._

## Introduction

#### What is this repository for? ####

The goal of this project is to detect speech from time-series signals. More precisely, distinguish between multi-channel signals which are recorded while speaking different words.

#### How do I get set up? ####
It is highly recommended to use a new environment. I used conda to create a new environment:

    conda create -n [name_of_the_environment] python=3.5 keras tensorflow
    source activate [name_of_the_environment]
    
Several python libraries are used; notably:

- For visualization purposes, `Matplotlib`, `Seaborn` and `Tensorboard` are used.
- For data manipulation, `numpy` is usually used. 
- For model creation, `Keras` (based on `TensorFlow`) is used. 
- For evaluation, in addition to `Keras`, `Scikit-learn` is used.
- For storing models and data, `hdf5` and `json` are used. 

Use pip to install all the requirement from the requirements.txt.
    
    pip install -r requirements.txt

_(Note: Don't use sudo here.)_

## Dataset

I had access to multi-channel signals from multiple subjects for a number of words. The total size of the dataset was just over 10,000.  

## Walk through the process

The process contains 3 steps:

#### First, edit the config file
Edit the config.json file in the config folder to tune the parameters for conversion and training, set the source and destination folders, and set visualization parameters. The readme file in the config folder contains details about the config.json file.

### Second, convert the raw time-series to MFCCs
Run the python code in the conversion directory to convert the raw signals to their MFCC representations and save them to the destination directory:

    python convert_to_mfcc.py 

### Third, run learning algorithms over the data-set
Run the python code in the models directory:

    python run_classifiers.py

This script will take the machine learning architecture from the models.py file, depending on the value of "model" in the config.json file from step 1. Then it will run the training step on the training dataset. For each training epoch, depending on the maximum performance of the classifier on the validation dataset, the script will store the checkpoint in the models/checkpoints directory. After training, it will load the "best" checkpoint from the directory and run the evaluation process on the testing dataset. The training logs for loss and other metrics are saved in the models/logs directory. After the evaluation, the confusion matrix will be saved in the models/confusion_matrix directory, and the top-5 predictions will be saved in the model_polts. These generated files can be tracked by the timestamp of associated with each specific runs. 

##### Optional step: the python script in the visualization folder can be run for visualization purposes. 

##### Note: the python version used is python 3.5.

## Results

For all the subjects combined, I obtained accuracies over 70%. Whereas, after tuning for specific subjects, I obtained accuracies up to 87%.

## References used:
1. How Hearing Works: http://www.medel.com/us/how-hearing-works/
2. Mel Filter Bank: http://www.ee.columbia.edu/ln/rosa/doc/HTKBook21/node54.html
3. MFCC Feature Extraction: http://recognize-speech.com/feature-extraction/mfcc
4. Noise generation:  https://raw.githubusercontent.com/python-acoustics/python-acoustics/master/acoustics/generator.py
5. Librosa library: https://github.com/librosa/librosa
6. Voice Recognition Algorithms using Mel Frequency Cepstral Coefficient (MFCC): https://arxiv.org/abs/1003.4083
