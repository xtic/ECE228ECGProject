## ECE228ECGProject

#Changelog:

05.16.2020 - Direct link to datasets: https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698/1 - Uploaded an example .mat file for using the file Python - Daniel Valencia

05.17.2020 - Added jupyter notebook to show how to read a file, parse it, and split the data to create chunks of data and their corresponding MI markers (as given in the data description). - Daniel Valencia

05.21.2020 - Added jupyter notebook that uses an LSTM-based RNN to perform time-series prediction of MI tasks. - Daniel Valencia

05.26.2020 - Added two new jupyter notebooks that use a publically available implementation of CSP to generate the transformation filters that are then applied on the data. The notebook 'LSTM_Test_TwoClass_wCSP' is an extension of the previous 'LSTM_Test_TwoClass' notebook but it adds CSP and shows the performance of the same LSTM model. The new notebook 'LSTM_Test_5F' shows the testing results of the same sized LSTM on the Five Finger dataset, both with and without CSP. Using CSP shows performance increases in the 30% range for 5F, and smaller (~5-10%) for the CLA dataset. - Daniel Valencia

05.30.2020 - Added 'LSTM_Test_ThreeClass_wCSP.ipynb' which tests for left, right, and no signals

6.01.2020  - Added 'Preprocessing_Denoising.ipynb' which cleans EEG data based on wavelet-BSS algorithm. - Nick Wong

06.02.2020 - Added LSTM_5F_All_Single, LSTM_CLA_All_Single, and LSTM_HaLT_All_Single notebooks. These will iterate through all of the datasets for that particular type, train a different network for each .mat file, and produce a CSV with the results of the LSTMs results with and without CSP as a pre-processing step. Also added the directory Results_LSTM to show results of the LSTM trained in the notebooks above and gives a relative performance measure. It lists the results for each .mat file and shows the accuracy of the model on the training and test subsets. - Daniel Valencia

6.03.20  - Added 'Preprocessing_LSTM.ipnyb' which uses the cleaned EEG data via wavelet-BSS for the LSTM.  Dataset chosen was LR dataset but modified labels to 6 labels based on the signal change.  -Nick Wong

06.09.2020 - Added 'LSTM_Final_Notebook.ipynb', which allows the user to load all of the datasets for a particular paradigm and combine them into a single test set for all of the subjects together. Testing results have improved greatly. Also added a helper file 'helperFunctions.py' to help consolidate some commonly used code. - Daniel Valencia


# Submission Notes
How to run the final notebooks, LSTM_Final_Notebook and SVM_Final_Notebook.
1. First, you need to make sure that you have downloaded the dataset to your local machine after you've cloned this Git repository. Use the following two commands from a linux shell to obtain the datasets.

	wget --content-disposition https://ndownloader.figshare.com/files/{9636634..9636400}

	wget --content-disposition https://ndownloader.figshare.com/files/{12400370..12400412}

2. Next, make sure you have all of the python libraries installed. Namely, you will need tensorflow, matplotlib, numpy, scikit learn, PyWavelets, and SciPy.

3. Notes when running the LSTM_Final_Notebook and SVM_Final_Notebook:

	When you run this notebook, you will be prompted for the following things in this order:

	3.1. Enter the desired paradigm you wish to test. Valid options are 5F, HALT, or CLA.

	3.2. Enter the relative path to the datasets. This should be the path to the directory that contains the downloaded dataset in step 1 above. For example, your relative path might be ../../../datasets/5F_Data/. Make sure to include the forward slash at the end of your path directory.

	After you enter these details, the notebook will run on its own. First it will parse the datasets and perform any pre-processing, then it will create the model and train/evaluate it using no pre-processing, CSP pre-processing, BSS. You can disable or change the dropout percentage by changing the lines 'enableDropout=True' and 'dropoutPercentage=0.3', respectively. These values do nothing for the SVM notebook.

Note: There are a number of other notebooks used during the development of the two final notebooks. Feel free to check them out, but we cannot guarantee that they will run properly. We have left them available for entirety. One notebook that might be interesting to you is 'Pre-processing_Denoising.ipynb', which details and shows how the Wavelet-BSS algorithm works. It is nicely formatted and makes it clear the results of Wavelet-BSS on the raw data.

