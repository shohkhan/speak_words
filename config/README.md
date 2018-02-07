## This file contains details about the config_nn.json file
### The python files models/run_classifier.py and conversion/convert_to_mfcc.py reads the configurations from this file

**labels**: The array containing the vocabulary of words that will be included in the experiment. Can be a sub-set of the total available words.

**training_subjects**: The subjects to be included for training. Can be a subset of the total available subjects.

**testing_subjects**: The subjects to be included for testing. Can be a subset of the total available subjects.

**available_subjects**: The subjects to be included in the data conversion process (to MFCCs). Can be a subset of the total available subjects.

**training**:
- reload_model: If true, after training the model, the checkpoint with the best performance on the validation set will be loaded for testing.
- data_folder_name: The folder to be included into the experiment (training, validation and testing).
- train_data_path: Python format to indicate which sub-folders to be included into training.
- test_data_path: Python format to indicate which sub-folders to be included into testing.
- val_data_path: Python format to indicate which sub-folders to be included into validation.
- learning_rate: Learning rate of the (Adam) optimizer.
- epochs: Number of epochs to be run.
- batch_size: The batch size considered from training.
- model_checkpoint: A string name of a checkpoint of a model intended to be loaded. Leave blank, if the model is to be trained from the scratch.
- load_checked: Flag to notify that whether a checkpoint mentioned in the model_checkpoint field is to be loaded. Leave false, if the model is to be trained from the scratch. 
- predict_only: True if only testing is to be done, no training.

**model**: The model to be considered for the experiments.

**available_models**: The available models available in the package. Will not be read by python code. Included for references purposes.

**data_path**: Python format of the path to the raw data.

**conversion**:
- test_fraction: Which fraction of the data will be included into the testing and validation.
- conversion_type: The format of the converted data. Only available now is mfcc.
- apply_stack: If true, the inputs from different channels will be stacked to make a multi channel input. 
- apply_interpolation: If true, inputs will be converted to MFCCs and then using linear interpolation, made to be the same size (useful for neural nets and classical regression algorithms). 
- apply_padding: If true, inputs will be converted to MFCCs and then using zero padding, made to be the same size (useful for neural nets and classical regression algorithms).
- skip_second: Every second sample will be skipped. (Not used)
- sampling_frequency: Sampling frequency of the input signal.
- highfreq: The highest frequency to be considered for MFCC conversion.
- numcep: Number of Mel frequency cepstral bands to be considered for conversion.
- nfilt: Number of filters (beginning from the lowest frequency band) to be considered.
- new_length: The new length after interpolation or padding
- minimum_data_size: Minimum sample size of the input signals are considered. (For sampling freq of 2000Hz, minimum_data_size of 200 indicates the minimum input size should be of 0.1 seconds)
- show_plot: Show visual representations while doing the conversion (see the code)
- noises: Which noises are to be included for data augmentation.
- noises_available: All available noises.
- noise_coefficients: The coefficients to be multiplied with the noise before adding to the original input signal.
- apply_shift: If true, left and right shifting will be done for data augmentation.
- apply_stretch": If true, input signal will be stretched (using an external library)

**all_labels**: For reference, all the available words in the vocabulary.