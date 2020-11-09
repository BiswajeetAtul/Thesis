# Thesis

## A COMPARISON BETWEEN DEEP LEARNING MODELS AND EXTREME LEARNING MACHINES FOR MULTI-LABEL TEXT CLASSIFICATION

We compare ELMs and Multi-Channel, Multi-Filter, Bi-LSTM, CNNs for the task of multi-label text classification.

## Folder Structure

* BERT:
Deals with code related to BERT embedding generation

* BiLSTM:
Has the code for Multi-Channel, Multi-Filter, Bi-LSTM, CNNs
  * BILSTM.ipynb has the basic code of the model
  * MultiChannel_CNN_BILSTM_FINAL.ipynb has the code which creates and runs all the models across hyperparameters
  * MultiChannelBiLSTMCNN.py has the class definition for Multi-Channel, Multi-Filter, Bi-LSTM, CNNs
  * Bi-LSTM_Evaluation.ipynb gives has the metrics and runtimes result visualization

* Dataset:
Stores the dataset folder

* EDA:
  * EDA.ipynb has the EDA data and insights

* ELM:
  * ELM_BERT.ipynb has the code for running ELM WITH BERT EMBEDDINGS
  * ELM_XlNET.ipynb has the code for running ELM WITH XLNET EMBEDDINGS
  * ELM_Model_Test_results.ipynb has the code for running ELM evaluation metrics

* Preprocessing:
  * Preprocessing.ipynb does all the preprocessing of the dataset

* XLNet:
has the code which generates the xlnet embeddings
