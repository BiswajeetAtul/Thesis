# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import prettytable
import tensorflow as tf
from keras import callbacks, initializers, optimizers, regularizers
from keras.layers import (LSTM, Bidirectional, Concatenate, Conv1D, Dense,
                          Dropout, Flatten, Input, MaxPooling1D,
                          TimeDistributed)
from keras.models import Model, Sequential
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             precision_recall_curve, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from keras.models import load_model


# %%
tf.__version__


# %%
import pandas as pd
import numpy as np
import datetime


# %%
if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


# %%
# Loading Tensorboard Extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# %%
# Metrics Calculator Function
def evaluate_model(real, predicted):
    accuracy = accuracy_score(real, predicted)
    hamLoss = hamming_loss(real, predicted)
    # element wise correctness
    term_wise_accuracy = np.sum(np.logical_not(
        np.logical_xor(real, predicted)))/real.size

    macro_precision = precision_score(real, predicted, average='macro')
    macro_recall = recall_score(real, predicted, average='macro')
    macro_f1 = f1_score(real, predicted, average='macro')

    micro_precision = precision_score(real, predicted, average='micro')
    micro_recall = recall_score(real, predicted, average='micro')
    micro_f1 = f1_score(real, predicted, average='micro')

    metricTable = prettytable.PrettyTable()
    metricTable.field_names = ["Metric", "Macro Value", "Micro Value"]
    metricTable.add_row(["Hamming Loss", "{0:.3f}".format(hamLoss), ""])
    metricTable.add_row(
        ["Term Wise Accuracy", "{0:.3f}".format(term_wise_accuracy), ""])

    metricTable.add_row(["Accuracy", "{0:.3f}".format(accuracy), ""])
    metricTable.add_row(["Precision", "{0:.3f}".format(
        macro_precision), "{0:.3f}".format(micro_precision)])
    metricTable.add_row(["Recall", "{0:.3f}".format(
        macro_recall), "{0:.3f}".format(micro_recall)])
    metricTable.add_row(
        ["F1-measure", "{0:.3f}".format(macro_f1), "{0:.3f}".format(micro_f1)])

    print(metricTable)


# %%
#Helper Functions
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
  
def initial_boost(epoch):
    if epoch==0: return float(8.0)
    elif epoch==1: return float(4.0)
    elif epoch==2: return float(2.0)
    elif epoch==3: return float(1.5)
    else: return float(1.0)

def step_cyclic(epoch):
    try:
        l_r, decay = 1.0, 0.0001
        if epoch%33==0:multiplier = 10
        else:multiplier = 1
        rate = float(multiplier * l_r * 1/(1 + decay * epoch))
        #print("Epoch",epoch+1,"- learning_rate",rate)
        return rate
    except Exception as e:
        print("Error in lr_schedule:",str(e))
        return float(1.0)

# Loading Data

# %%
mpstDF= pd.read_csv("mpst.csv")
mpstDF


# %%
# Data Split Function
def get_partition_Embeddings(x_t1,x_t2,y,df,partition_nm):
    _df=df[df["split"]==partition_nm]
    index_list=list(_df.index)
    temp_array_x_t1=[]
    temp_array_x_t2=[]
    temp_array_y=[]
    for index in index_list:
        temp_array_x_t1.append(x_t1[index,:])
        temp_array_x_t2.append(x_t2[index,:])
        temp_array_y.append(y[index,:])
    temp_array_x_t1=np.array(temp_array_x_t1)
    temp_array_x_t2=np.array(temp_array_x_t2)
    temp_array_y=np.array(temp_array_y)
    return temp_array_x_t1,temp_array_x_t2, temp_array_y


# %%
# LOADING BERT  EMBEDDINGS
bert_embedding=np.load(r"D:\CodeRepo\Thesis\Thesis\XLNet\xl_embeddings.npz")
# LOADING XLnet EMBEDDINGS
xl_embedding=np.load(r"D:\CodeRepo\Thesis\Thesis\XLNet\xl_embeddings.npz")
# LOADING LABELS Y
label_values=np.load(r"D:\CodeRepo\Thesis\Thesis\EDA\Y.npz")


# %%
# BERT EMBEDDINGS T1 & T2
type1_BERT_Embeddings=bert_embedding["t1"]
type2_BERT_Embeddings=bert_embedding["t2"]
# XLNet EMBEDDINGS T1 & T2
type1_XL_Embeddings=xl_embedding["t1"]
type2_XL_Embeddings=xl_embedding["t2"]
# LABLES Y
label_values=label_values["arr_0"]


# %%
# BERT

# FOR TRAIN
type1_BERT_Embeddings_Train,type2_BERT_Embeddings_Train,BERT_label_values_Train=get_partition_Embeddings(type1_BERT_Embeddings,type2_BERT_Embeddings,label_values,mpstDF,"train")
# FOR VALIDATION
type1_BERT_Embeddings_Val,type2_BERT_Embeddings_Val,BERT_label_values_Val=get_partition_Embeddings(type1_BERT_Embeddings,type2_BERT_Embeddings,label_values,mpstDF,"val")
# FOR TEST
type1_BERT_Embeddings_Test,type2_BERT_Embeddings_Test,BERT_label_values_Test=get_partition_Embeddings(type1_BERT_Embeddings,type2_BERT_Embeddings,label_values,mpstDF,"test")


# %%
# XLNET

# FOR TRAIN
type1_XL_Embeddings_Train,type2_XL_Embeddings_Train,XLNET_label_values_Train=get_partition_Embeddings(type1_XL_Embeddings,type2_XL_Embeddings,label_values,mpstDF,"train")
# FOR VALIDATION
type1_XL_Embeddings_Val,type2_XL_Embeddings_Val,XLNET_label_values_Val=get_partition_Embeddings(type1_XL_Embeddings,type2_XL_Embeddings,label_values,mpstDF,"val")
# FOR TEST
type1_XL_Embeddings_Test,type2_XL_Embeddings_Test,XLNET_label_values_Test=get_partition_Embeddings(type1_XL_Embeddings,type2_XL_Embeddings,label_values,mpstDF,"test")


# HAVING A LOOK AT THE SHAPES OF EACH PARTITION CREATED:

# %%
print("SHAPES OF EACH PARTITION _ BERT\n")
print("type1_BERT_Embeddings_Train.shape: ", type1_BERT_Embeddings_Train.shape)
print("type2_BERT_Embeddings_Train.shape: ", type2_BERT_Embeddings_Train.shape)
print("BERT_label_values_Train.shape: "    , BERT_label_values_Train.shape)

print("type1_BERT_Embeddings_Val.shape: ", type1_BERT_Embeddings_Val.shape)
print("type2_BERT_Embeddings_Val.shape: ", type2_BERT_Embeddings_Val.shape)
print("BERT_label_values_Val.shape: "    , BERT_label_values_Val.shape)

print("type1_BERT_Embeddings_Test.shape: ", type1_BERT_Embeddings_Test.shape)
print("type2_BERT_Embeddings_Test.shape: ", type2_BERT_Embeddings_Test.shape)
print("BERT_label_values_Test.shape: "    , BERT_label_values_Test.shape)

print("SHAPES OF EACH PARTITION _ XLNET\n")

print("type1_XL_Embeddings_Train.shape: ", type1_XL_Embeddings_Train.shape)
print("type2_XL_Embeddings_Train.shape: ", type2_XL_Embeddings_Train.shape)
print("XLNET_label_values_Train.shape: " , XLNET_label_values_Train.shape)

print("type1_XL_Embeddings_Val.shape: ", type1_XL_Embeddings_Val.shape)
print("type2_XL_Embeddings_Val.shape: ", type2_XL_Embeddings_Val.shape)
print("XLNET_label_values_Val.shape: " , XLNET_label_values_Val.shape)

print("type1_XL_Embeddings_Test.shape: ", type1_XL_Embeddings_Test.shape)
print("type2_XL_Embeddings_Test.shape: ", type2_XL_Embeddings_Test.shape)
print("XLNET_label_values_Test.shape: " , XLNET_label_values_Test.shape)

print("SHAPES OF EACH PARTITION _ BERT\n")
print("type1_BERT_Embeddings_Train.: ", type1_BERT_Embeddings_Train)
print("type2_BERT_Embeddings_Train.: ", type2_BERT_Embeddings_Train)
print("BERT_label_values_Train.: "    , BERT_label_values_Train)

print("type1_BERT_Embeddings_Val.: ", type1_BERT_Embeddings_Val)
print("type2_BERT_Embeddings_Val.: ", type2_BERT_Embeddings_Val)
print("BERT_label_values_Val.: "    , BERT_label_values_Val)

print("type1_BERT_Embeddings_Test.shape: ", type1_BERT_Embeddings_Test)
print("type2_BERT_Embeddings_Test.shape: ", type2_BERT_Embeddings_Test)
print("BERT_label_values_Test.shape: "    , BERT_label_values_Test)

print("SHAPES OF EACH PARTITION _ XLNET\n")

print("type1_XL_Embeddings_Train: ", type1_XL_Embeddings_Train)
print("type2_XL_Embeddings_Train: ", type2_XL_Embeddings_Train)
print("XLNET_label_values_Train: " , XLNET_label_values_Train)

print("type1_XL_Embeddings_Val: ", type1_XL_Embeddings_Val)
print("type2_XL_Embeddings_Val: ", type2_XL_Embeddings_Val)
print("XLNET_label_values_Val: " , XLNET_label_values_Val)

print("type1_XL_Embeddings_Test: ", type1_XL_Embeddings_Test)
print("type2_XL_Embeddings_Test: ", type2_XL_Embeddings_Test)
print("XLNET_label_values_Test: " , XLNET_label_values_Test)


# %%
# function to reshape my input
def tensor_reshape(input, timestep, features):
  reshaped= input.reshape(type1_BERT_Embeddings_Train.shape[0], timestep, features)
  return reshaped


# %%
"""
STARTING THE MODEL
possible combination in inputs:
(timestep, features)
- (1, 768)
- (2, 384)
- (3, 256)
- (4, 192)
- (6, 128)
- (12, 64)

"""
# ## MODEL (1, 768):
# 

# %%
INPUT_SHAPE =(1,768)
EM_L_F_UNITS= 768
EM_L_T_UNITS= 768
# LEFT CHANNEL
LSTM_1F_UNITS= 768
LSTM_1T_UNITS= 768

CONV_2_FILTER= 24
CONV_2_KERNEL= 2
CONV_3_FILTER= 24
CONV_3_KERNEL= 2
CONV_5_FILTER= 24
CONV_5_KERNEL= 2
CONV_6_FILTER= 24
CONV_6_KERNEL= 2
CONV_8_FILTER= 24
CONV_8_KERNEL= 2

# RIGHT CHANNEL 
CONV_4F_FILTERS = 12
CONV_4F_KERNEL = 2
CONV_4T_FILTERS = 12
CONV_4T_KERNEL = 2

CONV_3F_FILTERS = 12
CONV_3F_KERNEL = 2
CONV_3T_FILTERS = 12
CONV_3T_KERNEL = 2

CONV_2F_FILTERS = 12
CONV_2F_KERNEL = 2
CONV_2T_FILTERS = 12
CONV_2T_KERNEL = 2

LSTM_2_C_L_UNITS = 128

# %%
class BalanceNet(object):
    """
    docstring
    """
    def __init__(self, parameter_list):
        """
        CONSTRUCTOR
        """
        inp_layer = Input(shape=INPUT_SHAPE, dtype="float32")
        embedding_layer_frozen=TimeDistributed(Dense(EM_L_F_UNITS,  trainable= False))(inp_layer)
        embedding_layer_train= TimeDistributed(Dense(EM_L_T_UNITS,  trainable= True))(inp_layer)
        print("inp_layer",inp_layer.shape)
        print("embedding_layer_frozen",embedding_layer_frozen.shape)
        print("embedding_layer_train", embedding_layer_train.shape)
        
        # ### LSTM TO CNN LEFT CHANNEL
        # ### LEFT LSTM PART

        l_lstm_1f =Bidirectional(LSTM(LSTM_1F_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(embedding_layer_frozen)
        l_lstm_1t =Bidirectional(LSTM(LSTM_1T_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(embedding_layer_train)

        l_lstm1 = Concatenate(axis=1)([l_lstm_1f, l_lstm_1t])
        print("l_lstm_1f",l_lstm_1f.shape)
        print("l_lstm_1t",l_lstm_1t.shape)
        print("l_lstm1", l_lstm1.shape)
        
        # ### LEFT CNN PART
        conv_1=[]
        l_conv_2 = Conv1D(filters=CONV_2_FILTER, kernel_size=CONV_2_KERNEL, activation='relu')(l_lstm1)
        l_conv_2 = Dropout(0.3)(l_conv_2)
        print("l_conv_2",l_conv_2.shape)

        conv_1.append(l_conv_2)

        if CONV_3_FILTER!=0:
            l_conv_3 = Conv1D(filters=CONV_3_FILTER, kernel_size=CONV_3_KERNEL, activation='relu')(l_lstm1)
            l_conv_3 = Dropout(0.3)(l_conv_3)
            print("l_conv_3",l_conv_3.shape)

            conv_1.append(l_conv_3)


        if CONV_5_FILTER!=0:
            l_conv_5 = Conv1D(filters=CONV_5_FILTER, kernel_size=CONV_5_KERNEL, activation='relu')(l_lstm1)
            l_conv_5 = Dropout(0.3)(l_conv_5)
            print("l_conv_5",l_conv_5.shape)

            conv_1.append(l_conv_5)

        
        if CONV_6_FILTER!=0:
            l_conv_6 = Conv1D(filters=CONV_6_FILTER, kernel_size=CONV_6_KERNEL, kernel_regularizer=regularizers.l2(0.001) ,activation='relu')(l_lstm1)
            l_conv_6 = Dropout(0.3)(l_conv_6)
            print("l_conv_6",l_conv_6.shape)

            conv_1.append(l_conv_6)

        if CONV_8_FILTER!=0:
            l_conv_8 = Conv1D(filters=CONV_8_FILTER, kernel_size=CONV_8_KERNEL, kernel_regularizer=regularizers.l2(0.001) ,activation='relu')(l_lstm1)
            l_conv_8 = Dropout(0.3)(l_conv_8)
            conv_1.append(l_conv_8)
            print("l_conv_8",l_conv_8.shape)



        l_lstm_c = Concatenate(axis =1)(conv_1)
        print("l_lstm_c", l_lstm_c.shape)
        
        # END LEFT CHANNEL

        # RIGHT CHANNEL




# %%
inp_layer = Input(shape=INPUT_SHAPE, dtype="float32")
embedding_layer_frozen=TimeDistributed(Dense(EM_L_F_UNITS,  trainable= False))(inp_layer)
embedding_layer_train= TimeDistributed(Dense(EM_L_T_UNITS,  trainable= True))(inp_layer)
print("inp_layer",inp_layer.shape)
print("embedding_layer_frozen",embedding_layer_frozen.shape)
print("embedding_layer_train",embedding_layer_train.shape)


# ### LSTM TO CNN LEFT CHANNEL

# ### LEFT LSTM PART

# %%
l_lstm_1f =Bidirectional(LSTM(LSTM_1F_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(embedding_layer_frozen)
l_lstm_1t =Bidirectional(LSTM(LSTM_1T_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(embedding_layer_train)

l_lstm1 = Concatenate(axis=1)([l_lstm_1f, l_lstm_1t])
print("l_lstm_1f",l_lstm_1f.shape)
print("l_lstm_1t",l_lstm_1t.shape)
print("l_lstm1",l_lstm1.shape)


# ### LEFT CNN PART

# %%

l_conv_2 = Conv1D(filters=CONV_2_FILTER, kernel_size=CONV_2_KERNEL, activation='relu')(l_lstm1)
l_conv_2 = Dropout(0.3)(l_conv_2)

# l_conv_3 = Conv1D(filters=CONV_3_FILTER, kernel_size=CONV_3_KERNEL, activation='relu')(l_lstm1)
# l_conv_3 = Dropout(0.3)(l_conv_3)

# l_conv_5 = Conv1D(filters=CONV_5_FILTER, kernel_size=CONV_5_KERNEL, activation='relu')(l_lstm1)
# l_conv_5 = Dropout(0.3)(l_conv_5)

l_conv_6 = Conv1D(filters=CONV_6_FILTER, kernel_size=CONV_6_KERNEL, kernel_regularizer=regularizers.l2(0.001) ,activation='relu')(l_lstm1)
l_conv_6 = Dropout(0.3)(l_conv_6)

# l_conv_8 = Conv1D(filters=CONV_8_FILTER, kernel_size=CONV_8_KERNEL, kernel_regularizer=regularizers.l2(0.001) ,activation='relu')(l_lstm1)
# l_conv_8 = Dropout(0.3)(l_conv_8)

# conv_1 =[l_conv_6, l_conv_5, l_conv_8, l_conv_2, l_conv_3 ]
conv_1 =[l_conv_6, l_conv_2 ]

l_lstm_c = Concatenate(axis =1)(conv_1)
print("l_conv_2",l_conv_2.shape)
print("l_conv_6",l_conv_6.shape)
print("l_lstm_c",l_lstm_c.shape)


# ### CNN TO LSTM RIGHT CHANNEL

# ### RIGHT CNN PART

# %%
# CNN TO LSTM
l_conv_4f = Conv1D(filters= CONV_4F_FILTERS, kernel_size=CONV_4F_KERNEL, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(embedding_layer_frozen)
l_conv_4f = Dropout(0.3)(l_conv_4f)

l_conv_4t = Conv1D(filters= CONV_4T_FILTERS, kernel_size=CONV_4T_KERNEL, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(embedding_layer_train)
l_conv_4t = Dropout(0.3)(l_conv_4t)

l_conv_3f = Conv1D(filters= CONV_3F_FILTERS, kernel_size=CONV_3F_KERNEL, activation='relu')(embedding_layer_frozen)
l_conv_3f = Dropout(0.3)(l_conv_3f)

l_conv_3t = Conv1D(filters= CONV_3T_FILTERS, kernel_size=CONV_3T_KERNEL, activation='relu')(embedding_layer_train)
l_conv_3t = Dropout(0.3)(l_conv_3t)

# l_conv_2f = Conv1D(filters= CONV_2F_FILTERS, kernel_size=CONV_2F_KERNEL, activation='relu')(embedding_layer_frozen)
# l_conv_2f = Dropout(0.3)(l_conv_2f)

# l_conv_2t = Conv1D(filters= CONV_2T_FILTERS, kernel_size=CONV_2T_KERNEL, activation='relu')(embedding_layer_train)
# l_conv_2t = Dropout(0.3)(l_conv_2t)

# conv_2 = [l_conv_4f, l_conv_4t, l_conv_3f, l_conv_3t, l_conv_2f, l_conv_2t]
conv_2 = [l_conv_4f, l_conv_4t, l_conv_3f, l_conv_3t, ]

l_merge_2 = Concatenate(axis=1)(conv_2)
print("l_conv_4f",l_conv_4f.shape)
print("l_conv_4t",l_conv_4t.shape)
print("l_conv_3f",l_conv_3f.shape)
print("l_conv_3t",l_conv_3t.shape)
print("l_merge_2",l_merge_2.shape)


# ### RIGHT LSTM PART

# %%
l_c_lstm = Bidirectional(LSTM(LSTM_2_C_L_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(l_merge_2)
print("l_c_lstm",l_c_lstm.shape)


# ###  FINAL CONCAT

# %%

l_merge = Concatenate(axis=1)([l_lstm_c, l_c_lstm])
l_pool = MaxPooling1D(4)(l_merge)
l_drop = Dropout(0.5)(l_pool)
l_flat = Flatten()(l_drop)
l_dense = Dense(128, activation='sigmoid')(l_flat)
preds= Dense(71, activation='sigmoid')(l_dense)


# %%



