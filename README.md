## ECE228ECGProject

#Changelog:

05.16.2020 - Direct link to datasets: https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698/1 - Uploaded an example .mat file for using the file Python - Daniel Valencia

05.17.2020 - Added jupyter notebook to show how to read a file, parse it, and split the data to create chunks of data and their corresponding MI markers (as given in the data description). - Daniel Valencia

05.21.2020 - Added jupyter notebook that uses an LSTM-based RNN to perform time-series prediction of MI tasks. - Daniel Valencia

05.26.2020 - Added two new jupyet notebooks that use a publically available implementation of CSP to generate the transformation filters that are then applied on the data. The notebook 'LSTM_Test_TwoClass_wCSP' is an extension of the previous 'LSTM_Test_TwoClass' notebook but it adds CSP and shows the performance of the same LSTM model. The new notebook 'LSTM_Test_5F' shows the testing results of the same sized LSTM on the Five Finger dataset, both with and without CSP. Using CSP shows performance increases in the 30% range for 5F, and smaller (~5-10%) for the CLA dataset. - Daniel Valencia

05.30.2020 - Added 'LSTM_Test_ThreeClass_wCSP.ipynb' which tests for left, right, and no signals

6.01.2020  - Added 'Preprocessing_Denoising.ipynb' which cleans EEG data based on wavelet-BSS algorithm.
