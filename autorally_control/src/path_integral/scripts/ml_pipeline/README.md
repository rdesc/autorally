# ML pipeline
The pipeline consists of 3 standard components: data preprocessing, model training, and model testing/evaluation. The
pipeline is initiated via the __trainer.py__ script (i.e. `python trainer.py`) and is configured via the _config.yml_ parameter file. 

### Sample pipeline output
[Here](https://drive.google.com/drive/folders/18RyBF3rOT8EYqhjZNGO42cna23OOXpTX?usp=sharing) and [here](https://drive.google.com/drive/folders/1ySzDX9JWf1Y3PGeK6iSBXYrupQeQwVtM?usp=sharing) 
are links containing sample outputs using this pipeline. The former contains a preprocessed test dataset, preprocessing plots, and original rostopic .csv + rosbag files.
The latter contains preprocessing files for the training dataset, model training files, and model testing files. The models in this run were tested with the test dataset
mentioned earlier. 

## Shared parameters
Parameters shared across pipeline components.
- __run_name__ - name of the directory to create which will store the outputs of different phases associated with this run
- __standardize_data__ - option to standardize data via the [sklearn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
Note, MPPI expects the outputs of the model to be non-scaled. If the model is trained on scaled features and labels, then
the outputs need to be transformed back to their original scaling before updating the states. However, the MPPI code currently does 
not contain any logic to handle this. A potential fix could be modifying the first and final layers of the network to include
the linear scaling.
- __state_cols__ - the names of the state columns 
- __ctrl_cols__ - the names of the control columns
- __feature_cols__ - the names of the feature columns
- __label_cols__ - the names of the label columns
- __cuda_device__ - the ID of the CUDA device
- __nn_layers__ - list which specifies the number of nodes for each layer (default is [6, 32, 32, 4])

## Data preprocessing
Loads up a rosbag file, creates a separate .csv file for each of the specified rostopics to extract from the rosbag, and
applies a bunch of standard data preprocessing. The final output of this phase is an aggregated .csv file containing the preprocessed dataset.

### Parameters
- __preprocess_data__ - set to True to initiate the data preprocessing phase
- __rosbag_filepath__ - the file path of the rosbag file
- __make_preprocessing_plots__ - option to plot the preprocessing plots
- __total_data__ - total amount of data to keep in seconds, will keep all data if left blank
- __final_file_name__ - name of the final aggregated .csv file (e.g. train_data.csv for the training data and test_data.csv for the test data)
- __topics__ - arbitrary sized list specifying the rostopics to extract from the rosbag file and the preprocessing to apply to the topic data
    - __name__ - name of the rostopic
    - __col_mapper__ - arbitrary sized dictionary where keys are the original column names and the values are the new columns names
    (NOTE: pandas will add a '.' and a number for duplicate columns when it loads a .csv topic file)
    - __resample__ - resamples the data specified by the __cols__ parameter at a resulting sample rate of __upsampling_factor__ / __downsampling_factor__ * original sample rate
    - __filename__ - csv file name for the preprocessed topic data
    - __quaternion_to_euler__ - optional arg to convert the quaternion data specified by __x__, __y__, __z__, __w__ to euler angles
    - __compute_derivatives__ - optional arg to compute derivatives of the data specified by __cols__
    - __trunc__ - optional arg to truncate the data specified by __cols__

## Model training
Model training is done with the [PyTorch library](https://pytorch.org/). This phase sets up a feedforward neural network.
By default the activation function is the [Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)
non-linearity and the loss function is the [Smooth L1 Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html).
The validation data consists of 20% of the overall training dataset, the rest is the training data. Both the individual loss components and 
the overall loss for each epoch are recorded and plotted at the end of training.

### Parameters
- __train_model__ - set to True to initialize the model training phase
- __results_dir__ - the directory to store results from the training phase
- __model_dir_name__ - the name of directory which will contain the model and loss monitoring plots
- __training_data_path__ - the path of the training data
- __train_data_fraction__ - fraction of the overall training dataset to use
- __loss_weights__ - the weights for the loss components
- __epochs__ - number of epochs
- __batch_size__ - batch size
- __lr__ - learning rate
- __weight_decay__ - [weight decay](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)


## Model testing
The trained network performs inference on a test dataset. The following are the two types of errors that are computed:
- __Instantaneous error__ - A sample is fed into the network and then the model output is compared to the ground truth by computing the difference.
The plot _inst_error_hist.pdf_ are histograms of the signed instantaneous errors for each of the network output components.
- __Multi step error__ - The test dataset is split into a specified number of batches. For each batch, the initial state and controls
are fed into the network. Next, the output of the network is used to compute the next state which is then fed back into the network along with 
the next set of controls from the test data. This repeats for all samples of the batch. 

### Parameters
- __test_model__ - set to True to initialize the model testing phase
- __model_dir_path__ - the directory path containing the _model.pt_ file
- __test_data_path__ - the path of the test data
- __test_data_fraction__ - fraction of the test data to use
- __state_dim__ - the size of the state space
- __time_horizon__ - the amount of time in seconds to propagate dynamics for (i.e. the size of the batches when the test dataset is split up)
- __scaler_paths__ - the paths which contain the scalers if model was trained on standardized data

# model_vehicle_dynamics.py
This script compares the output of the neural network to ODE integration. A shared set of initial conditions are specified as well as the 
steering and throttle controls to be applied at each time step. The output of the network is used to update the next state which is then fed back
into the network. This repeats continuously for all time steps up until some time horizon. Sample outputs from this script 
can be found [here](https://drive.google.com/drive/folders/1umg91lUWqW036agO9n1Zxo41iHHsKDhm?usp=sharing).