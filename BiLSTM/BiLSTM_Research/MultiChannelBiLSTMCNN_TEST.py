# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

# from IPython import get_ipython


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



tf.__version__



import pandas as pd
import numpy as np
import datetime, time



# if tf.test.gpu_device_name(): 
#     print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")



# Loading Tensorboard Extension
# get_ipython().run_line_magic('load_ext', 'tensorboard')



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



#Helper Functions
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
  
# def initial_boost(epoch):
#     if epoch==0: return float(8.0)
#     elif epoch==1: return float(4.0)
#     elif epoch==2: return float(2.0)
#     elif epoch==3: return float(1.5)
#     else: return float(1.0)
def initial_boost(epoch, lr):
    # print("THE LEARNING RATE IS: ",lr)
    if epoch < 10:
        return 0.01
    else:
        return float(lr * tf.math.exp(-0.1))

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


# mpstDF= pd.read_csv("mpst.csv")
# mpstDF


# 
# # Data Split Function
# def get_partition_Embeddings(x_t1,x_t2,y,df,partition_nm):
#     _df=df[df["split"]==partition_nm]
#     index_list=list(_df.index)
#     temp_array_x_t1=[]
#     temp_array_x_t2=[]
#     temp_array_y=[]
#     for index in index_list:
#         temp_array_x_t1.append(x_t1[index,:])
#         temp_array_x_t2.append(x_t2[index,:])
#         temp_array_y.append(y[index,:])
#     temp_array_x_t1=np.array(temp_array_x_t1)
#     temp_array_x_t2=np.array(temp_array_x_t2)
#     temp_array_y=np.array(temp_array_y)
#     return temp_array_x_t1,temp_array_x_t2, temp_array_y


# 
# # LOADING BERT  EMBEDDINGS
# bert_embedding=np.load(r"D:\CodeRepo\Thesis\Thesis\XLNet\xl_embeddings.npz")
# # LOADING XLnet EMBEDDINGS
# xl_embedding=np.load(r"D:\CodeRepo\Thesis\Thesis\XLNet\xl_embeddings.npz")
# # LOADING LABELS Y
# label_values=np.load(r"D:\CodeRepo\Thesis\Thesis\EDA\Y.npz")


# 
# # BERT EMBEDDINGS T1 & T2
# type1_BERT_Embeddings=bert_embedding["t1"]
# type2_BERT_Embeddings=bert_embedding["t2"]
# # XLNet EMBEDDINGS T1 & T2
# type1_XL_Embeddings=xl_embedding["t1"]
# type2_XL_Embeddings=xl_embedding["t2"]
# # LABLES Y
# label_values=label_values["arr_0"]


# 
# # BERT

# # FOR TRAIN
# type1_BERT_Embeddings_Train,type2_BERT_Embeddings_Train,BERT_label_values_Train=get_partition_Embeddings(type1_BERT_Embeddings,type2_BERT_Embeddings,label_values,mpstDF,"train")
# # FOR VALIDATION
# type1_BERT_Embeddings_Val,type2_BERT_Embeddings_Val,BERT_label_values_Val=get_partition_Embeddings(type1_BERT_Embeddings,type2_BERT_Embeddings,label_values,mpstDF,"val")
# # FOR TEST
# type1_BERT_Embeddings_Test,type2_BERT_Embeddings_Test,BERT_label_values_Test=get_partition_Embeddings(type1_BERT_Embeddings,type2_BERT_Embeddings,label_values,mpstDF,"test")


# 
# # XLNET

# # FOR TRAIN
# type1_XL_Embeddings_Train,type2_XL_Embeddings_Train,XLNET_label_values_Train=get_partition_Embeddings(type1_XL_Embeddings,type2_XL_Embeddings,label_values,mpstDF,"train")
# # FOR VALIDATION
# type1_XL_Embeddings_Val,type2_XL_Embeddings_Val,XLNET_label_values_Val=get_partition_Embeddings(type1_XL_Embeddings,type2_XL_Embeddings,label_values,mpstDF,"val")
# # FOR TEST
# type1_XL_Embeddings_Test,type2_XL_Embeddings_Test,XLNET_label_values_Test=get_partition_Embeddings(type1_XL_Embeddings,type2_XL_Embeddings,label_values,mpstDF,"test")


# # HAVING A LOOK AT THE SHAPES OF EACH PARTITION CREATED:

# 
# print("SHAPES OF EACH PARTITION _ BERT\n")
# print("type1_BERT_Embeddings_Train.shape: ", type1_BERT_Embeddings_Train.shape)
# print("type2_BERT_Embeddings_Train.shape: ", type2_BERT_Embeddings_Train.shape)
# print("BERT_label_values_Train.shape: "    , BERT_label_values_Train.shape)

# print("type1_BERT_Embeddings_Val.shape: ", type1_BERT_Embeddings_Val.shape)
# print("type2_BERT_Embeddings_Val.shape: ", type2_BERT_Embeddings_Val.shape)
# print("BERT_label_values_Val.shape: "    , BERT_label_values_Val.shape)

# print("type1_BERT_Embeddings_Test.shape: ", type1_BERT_Embeddings_Test.shape)
# print("type2_BERT_Embeddings_Test.shape: ", type2_BERT_Embeddings_Test.shape)
# print("BERT_label_values_Test.shape: "    , BERT_label_values_Test.shape)

# print("SHAPES OF EACH PARTITION _ XLNET\n")

# print("type1_XL_Embeddings_Train.shape: ", type1_XL_Embeddings_Train.shape)
# print("type2_XL_Embeddings_Train.shape: ", type2_XL_Embeddings_Train.shape)
# print("XLNET_label_values_Train.shape: " , XLNET_label_values_Train.shape)

# print("type1_XL_Embeddings_Val.shape: ", type1_XL_Embeddings_Val.shape)
# print("type2_XL_Embeddings_Val.shape: ", type2_XL_Embeddings_Val.shape)
# print("XLNET_label_values_Val.shape: " , XLNET_label_values_Val.shape)

# print("type1_XL_Embeddings_Test.shape: ", type1_XL_Embeddings_Test.shape)
# print("type2_XL_Embeddings_Test.shape: ", type2_XL_Embeddings_Test.shape)
# print("XLNET_label_values_Test.shape: " , XLNET_label_values_Test.shape)

# print("SHAPES OF EACH PARTITION _ BERT\n")
# print("type1_BERT_Embeddings_Train.: ", type1_BERT_Embeddings_Train)
# print("type2_BERT_Embeddings_Train.: ", type2_BERT_Embeddings_Train)
# print("BERT_label_values_Train.: "    , BERT_label_values_Train)

# print("type1_BERT_Embeddings_Val.: ", type1_BERT_Embeddings_Val)
# print("type2_BERT_Embeddings_Val.: ", type2_BERT_Embeddings_Val)
# print("BERT_label_values_Val.: "    , BERT_label_values_Val)

# print("type1_BERT_Embeddings_Test.shape: ", type1_BERT_Embeddings_Test)
# print("type2_BERT_Embeddings_Test.shape: ", type2_BERT_Embeddings_Test)
# print("BERT_label_values_Test.shape: "    , BERT_label_values_Test)

# print("SHAPES OF EACH PARTITION _ XLNET\n")

# print("type1_XL_Embeddings_Train: ", type1_XL_Embeddings_Train)
# print("type2_XL_Embeddings_Train: ", type2_XL_Embeddings_Train)
# print("XLNET_label_values_Train: " , XLNET_label_values_Train)

# print("type1_XL_Embeddings_Val: ", type1_XL_Embeddings_Val)
# print("type2_XL_Embeddings_Val: ", type2_XL_Embeddings_Val)
# print("XLNET_label_values_Val: " , XLNET_label_values_Val)

# print("type1_XL_Embeddings_Test: ", type1_XL_Embeddings_Test)
# print("type2_XL_Embeddings_Test: ", type2_XL_Embeddings_Test)
# print("XLNET_label_values_Test: " , XLNET_label_values_Test)


# 
# # function to reshape my input
# def tensor_reshape(input, timestep, features):
#   reshaped= input.reshape(type1_BERT_Embeddings_Train.shape[0], timestep, features)
#   return reshaped



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

OUTPUT_DENSE_UNIT =128
OUTPUT_SIZE = 71


class BalanceNet(object):
    """
    docstring
    """
    def __init__(self,
                INPUT_SHAPE,
                EM_L_F_UNITS,
                EM_L_T_UNITS,
                LSTM_1F_UNITS,
                LSTM_1T_UNITS,
                CONV_2_FILTER,
                CONV_2_KERNEL,
                CONV_3_FILTER,
                CONV_3_KERNEL,
                CONV_5_FILTER,
                CONV_5_KERNEL,
                CONV_6_FILTER,
                CONV_6_KERNEL,
                CONV_8_FILTER,
                CONV_8_KERNEL,
                CONV_4F_FILTERS = 12,
                CONV_4F_KERNEL=2,
                CONV_4T_FILTERS = 12,
                CONV_4T_KERNEL = 2,
                CONV_3F_FILTERS = 12,
                CONV_3F_KERNEL = 2,
                CONV_3T_FILTERS = 12,
                CONV_3T_KERNEL = 2,
                CONV_2F_FILTERS = 12,
                CONV_2F_KERNEL = 2,
                CONV_2T_FILTERS = 12,
                CONV_2T_KERNEL=2,
                LSTM_2_C_L_UNITS=128,
                OUTPUT_DENSE_UNIT=128,
                OUTPUT_SIZE=71,
                optimizer_name= 'adam'):
        """__init__ [summary]

        Args:
            INPUT_SHAPE ([type]): [description]
            EM_L_F_UNITS ([type]): [description]
            EM_L_T_UNITS ([type]): [description]
            LSTM_1F_UNITS ([type]): [description]
            LSTM_1T_UNITS ([type]): [description]
            CONV_2_FILTER ([type]): [description]
            CONV_2_KERNEL ([type]): [description]
            CONV_3_FILTER ([type]): [description]
            CONV_3_KERNEL ([type]): [description]
            CONV_5_FILTER ([type]): [description]
            CONV_5_KERNEL ([type]): [description]
            CONV_6_FILTER ([type]): [description]
            CONV_6_KERNEL ([type]): [description]
            CONV_8_FILTER ([type]): [description]
            CONV_8_KERNEL ([type]): [description]
            CONV_4F_FILTERS (int, optional): [description]. Defaults to 12.
            CONV_4F_KERNEL (int, optional): [description]. Defaults to 2.
            CONV_4T_FILTERS (int, optional): [description]. Defaults to 12.
            CONV_4T_KERNEL (int, optional): [description]. Defaults to 2.
            CONV_3F_FILTERS (int, optional): [description]. Defaults to 12.
            CONV_3F_KERNEL (int, optional): [description]. Defaults to 2.
            CONV_3T_FILTERS (int, optional): [description]. Defaults to 12.
            CONV_3T_KERNEL (int, optional): [description]. Defaults to 2.
            CONV_2F_FILTERS (int, optional): [description]. Defaults to 12.
            CONV_2F_KERNEL (int, optional): [description]. Defaults to 2.
            CONV_2T_FILTERS (int, optional): [description]. Defaults to 12.
            CONV_2T_KERNEL (int, optional): [description]. Defaults to 2.
            LSTM_2_C_L_UNITS (int, optional): [description]. Defaults to 128.
            OUTPUT_DENSE_UNIT (int, optional): [description]. Defaults to 128.
            OUTPUT_SIZE (int, optional): [description]. Defaults to 71.
            optimizer_name (string, optional) : Possible values any valid tf otimizer, and 'adadelta'
        """
        self.__input_shape =INPUT_SHAPE
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
        # ### CNN TO LSTM RIGHT CHANNEL
        
        # ### RIGHT CNN PART
        conv_2 = []
        l_conv_4f = Conv1D(filters= CONV_4F_FILTERS, kernel_size=CONV_4F_KERNEL, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(embedding_layer_frozen)
        l_conv_4f = Dropout(0.3)(l_conv_4f)

        l_conv_4t = Conv1D(filters= CONV_4T_FILTERS, kernel_size=CONV_4T_KERNEL, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(embedding_layer_train)
        l_conv_4t = Dropout(0.3)(l_conv_4t)
        conv_2.append(l_conv_4f)
        conv_2.append(l_conv_4t)
        print("l_conv_4f",l_conv_4f.shape)
        print("l_conv_4t",l_conv_4t.shape)
        
        if CONV_3F_FILTERS!=0 and CONV_3T_FILTERS!=0:
            l_conv_3f = Conv1D(filters= CONV_3F_FILTERS, kernel_size=CONV_3F_KERNEL, activation='relu')(embedding_layer_frozen)
            l_conv_3f = Dropout(0.3)(l_conv_3f)

            l_conv_3t = Conv1D(filters= CONV_3T_FILTERS, kernel_size=CONV_3T_KERNEL, activation='relu')(embedding_layer_train)
            l_conv_3t = Dropout(0.3)(l_conv_3t)
            conv_2.append(l_conv_3f)
            conv_2.append(l_conv_3t)
            print("l_conv_3f",l_conv_3f.shape)
            print("l_conv_3t",l_conv_3t.shape)
        
        if CONV_2F_FILTERS!=0 and CONV_2T_FILTERS!=0:
            l_conv_2f = Conv1D(filters= CONV_2F_FILTERS, kernel_size=CONV_2F_KERNEL, activation='relu')(embedding_layer_frozen)
            l_conv_2f = Dropout(0.3)(l_conv_2f)

            l_conv_2t = Conv1D(filters= CONV_2T_FILTERS, kernel_size=CONV_2T_KERNEL, activation='relu')(embedding_layer_train)
            l_conv_2t = Dropout(0.3)(l_conv_2t)
            conv_2.append(l_conv_2f)
            conv_2.append(l_conv_2t)
            print("l_conv_2f",l_conv_2f.shape)
            print("l_conv_2t",l_conv_2t.shape)

        l_merge_2 = Concatenate(axis=1)(conv_2)
        print("l_merge_2",l_merge_2.shape)

        # ### RIGHT LSTM PART
        l_c_lstm = Bidirectional(LSTM(LSTM_2_C_L_UNITS, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(l_merge_2)
        print("l_c_lstm",l_c_lstm.shape)

        # FINAL MERGER
        l_merge = Concatenate(axis=1)([l_lstm_c, l_c_lstm])
        l_pool = MaxPooling1D(4)(l_merge)
        l_drop = Dropout(0.5)(l_pool)
        l_flat = Flatten()(l_drop)
        l_dense = Dense(OUTPUT_DENSE_UNIT, activation='sigmoid')(l_flat)
        preds= Dense(OUTPUT_SIZE, activation='sigmoid')(l_dense)

        self.model = Model(inp_layer, preds)
        
        
        adadelta = optimizers.Adadelta(lr=0.9, rho=0.90, epsilon=None, decay=0.001)
        lr_metric = get_lr_metric(adadelta)

        if optimizer_name == 'adadelta':
            optimizer_name = adadelta

        self.model.compile( loss='categorical_crossentropy',
                            optimizer=optimizer_name,
                            metrics=[   "acc",
                                        "categorical_accuracy",
                                        "top_k_categorical_accuracy",
                                        tf.keras.metrics.Precision(),
                                        tf.keras.metrics.Recall(),
                                        tf.keras.metrics.TruePositives(),
                                        tf.keras.metrics.TrueNegatives(),
                                        tf.keras.metrics.FalsePositives(),
                                        tf.keras.metrics.FalseNegatives()])
    def __reshape_input(self, input, batch_size, timestep, features):
        """__reshape_input [summary]

        Args:
            input [tensor, ndarray]: input array to be reshaped
            timestep ([integer]): No of timesteps to be considered
            features ([integer]): No of features to be considered in each time step

        Raises:
            ex: [ValueError] WHEN THE NDARRAY CANT BE RESHAPED

        Returns:
            [tensor, ndarray]: reshaped
        """
        try:
            reshaped= input.reshape((batch_size, timestep, features))
            print("Reshaped Into Shape: ", reshaped.shape)
            return reshaped
        except ValueError as ex:
            raise ex


    def fitModel(self, train_x, train_y, val_x, val_y, model_type, epochs_count, batch_count, time_dict, key):
        """fitModel [summary]

        Args:
            train_x ([ndarray]): [train_data_set]
            train_y ([ndarray]): [train_data_set labels]
            val_x ([ndarray]): [validation_data_set]
            val_y ([ndarray]): [validation_data_set labels]
            model_type ([string]): [--FORMAT: <embedding>_<embedding type>]
            epochs_count ([integer]): [determines what will be the epochs]
            batch_count ([integer]): [determines what will be the count of samples in each batch]
        """
        try:
            print("Training Progress for: "+model_type)
            dt_time =datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            log_dir = '/BiLSTM/logs/fit/' 
            log_hist = '/BiLSTM/logs/hist/'
            log_model = '/BiLSTM/logs/save/'
            modelSaveFileName = log_model+"best_model_"+ str(self.__input_shape[0])+ "_" +str(self.__input_shape[1]) + "_" + dt_time + "_" + model_type + ".h5"
            modelLogSaveFileName = log_hist+"model_log"+str(self.__input_shape[0])+ "_" +str(self.__input_shape[1]) + "_" + dt_time+ "_" + model_type + ".csv"


            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            earlyStopping_cb = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
            modelCheckpoint_cb = callbacks.ModelCheckpoint(modelSaveFileName, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
            lr_schedule_cb = callbacks.LearningRateScheduler(initial_boost)

            print("Reshaping inputs")
            X_train = self.__reshape_input(train_x, train_x.shape[0], self.__input_shape[0], self.__input_shape[1])
            X_val= self.__reshape_input(val_x, val_x.shape[0], self.__input_shape[0], self.__input_shape[1])
            print("Reshaped Successfully!")
            start_time= time.time()           
            # print(sum(np.isnan(X_train_)))
            model_log = self.model.fit(X_train, train_y,
                                    validation_data=(X_val, val_y),
                                    epochs=epochs_count,
                                    batch_size=batch_count,
                                    callbacks=[tensorboard_cb,earlyStopping_cb, modelCheckpoint_cb, lr_schedule_cb])
            end_time =time.time()
            log_time(key=key,start_time=start_time, end_time=end_time, phase="train",log_dict=time_dict)
            pd.DataFrame(model_log.history).to_csv(modelLogSaveFileName)
            
        except Exception as ex:
            print("Exception in fitModel")
            print(ex)

    def predictModel(self, test_x, test_y, threshold_list,time_dict,key):
        """predictModel predicts the model and evaluates it for various thresholds 

        Args:
            test_x ([ndarray]): test data set to be predicted
            test_y ([ndarray]): original labels set of the test dataset

        Returns:
            [ndarray]: [the predicted values from the model]
        """
        thresholds = threshold_list
        start_time= time.time()           
        
        _test_x = self.__reshape_input(test_x, test_x.shape[0], self.__input_shape[0], self.__input_shape[1])
        real = test_y
        predicted = self.model.predict(_test_x)
        for threshold in thresholds:
            print("At threshold of " + str(threshold))
            _predicted = predicted.copy()
            _predicted = np.apply_along_axis(np.vectorize(lambda x: 1 if x> threshold else 0) ,axis=1, arr=_predicted)

            accuracy = accuracy_score(real, _predicted)
            hamLoss = hamming_loss(real, _predicted)
            # element wise correctness
            term_wise_accuracy = np.sum(np.logical_not(np.logical_xor(real, _predicted)))/real.size

            macro_precision = precision_score(real, _predicted, average='macro')
            macro_recall = recall_score(real, _predicted, average='macro')
            macro_f1 = f1_score(real, _predicted, average='macro')

            micro_precision = precision_score(real, _predicted, average='micro')
            micro_recall = recall_score(real, _predicted, average='micro')
            micro_f1 = f1_score(real, _predicted, average='micro')

            metricTable = prettytable.PrettyTable()
            metricTable.field_names = ["Metric", "Macro Value", "Micro Value"]
            metricTable.add_row(["Hamming Loss", "{0:.3f}".format(hamLoss), ""])
            metricTable.add_row(["Term Wise Accuracy", "{0:.3f}".format(term_wise_accuracy), ""])
            metricTable.add_row(["Accuracy", "{0:.3f}".format(accuracy), ""])
            metricTable.add_row(["Precision", "{0:.3f}".format(macro_precision), "{0:.3f}".format(micro_precision)])
            metricTable.add_row(["Recall", "{0:.3f}".format(macro_recall), "{0:.3f}".format(micro_recall)])
            metricTable.add_row(["F1-measure", "{0:.3f}".format(macro_f1), "{0:.3f}".format(micro_f1)])

            count_1_as_1 = 0 #TP
            count_1_as_0 = 0 #FN
            count_0_as_1 = 0 #FP
            count_0_as_0 = 0 #TN
            total_real_1s = 0
            total_real_0s = 0
            for i in range(real.shape[0]):
                for j in range(real.shape[1]):
                    if real[i,j] == 1:
                        total_real_1s+=1
                        if _predicted[i,j]==1:
                            count_1_as_1 +=1
                        if _predicted[i,j]==0:
                            count_1_as_0 +=1
                    if real[i,j] == 0:
                        total_real_0s+=1
                        if _predicted[i,j]==1:
                            count_0_as_1 +=1
                        if _predicted[i,j]==0:
                            count_0_as_0 +=1
            TP = count_1_as_1
            FN = count_1_as_0
            FP = count_0_as_1
            TN = count_0_as_0
            print("count_1_as_1, TP",count_1_as_1)
            print("count_1_as_0, FN",count_1_as_0)
            print("count_0_as_1, FP",count_0_as_1)
            print("count_0_as_0, TN",count_0_as_0)
            print("total_real_1s",total_real_1s)
            print("total_real_0s",total_real_0s)
            # MY_Accuracy = (TP+TN)/(TP+FP+FN+TN)
            # MY_Precision = TP/(TP+FP)
            # MY_Recall = TP/(TP+FN)
            # MY_F1_Score = 2*(MY_Recall * MY_Precision) / (MY_Recall + MY_Precision)
            # print("MY_Accuracy",MY_Accuracy)
            # print("MY_Precision",MY_Precision)
            # print("MY_Recall",MY_Recall)
            # print("MY_F1_Score",MY_F1_Score)
            indepth_metricTable = prettytable.PrettyTable()
            indepth_metricTable.field_names = ["Metric", "Value"]
            indepth_metricTable.add_row(["True Positives, count_1_as_1", "{0:.0f}".format(TP)])
            indepth_metricTable.add_row(["False Negatives, count_1_as_0", "{0:.0f}".format(FN)])
            indepth_metricTable.add_row(["False Positives, count_0_as_1", "{0:.0f}".format(FP)])
            indepth_metricTable.add_row(["True Negatives, count_0_as_0", "{0:.0f}".format(TN)])
            indepth_metricTable.add_row(["Real 1s ", "{0:.0f}".format(total_real_1s)])
            indepth_metricTable.add_row(["Real 0s ", "{0:.0f}".format(total_real_0s)])
            print(metricTable)
            print(indepth_metricTable)
        end_time =time.time()
        log_time(key=key,start_time=start_time, end_time=end_time, phase="train",log_dict=time_dict)
        return predicted

def predictModel(model, test_x, test_y, threshold_list, input_shape):
    """predictModel [summary]
        Predicts the model and evaluates it for various thresholds
        USE THIS FOR MODELS WHICH HAVE BEEN LOADED FROM .h5 files. 
        
    Args:
        model ([Keras.Model]): [description]
        test_x ([ndarray]): test data set to be predicted
        test_y ([ndarray]): original labels set of the test dataset
        threshold_list ([list]): list of thresholds
        input_shape ([tupe len=2]): input shape (timestamps, features)

    Returns:
         [ndarray]: [the predicted values from the model]
    """


    thresholds = threshold_list
    
    _test_x = np.reshape(test_x, input_shape[0], input_shape[1])
    real = test_y
    predicted = model.predict(_test_x)
    for threshold in thresholds:
        print("At threshold of " + str(threshold))
        _predicted = predicted.copy()
        np.apply_along_axis(np.vectorize(lambda x: 1 if x> threshold else 0) ,axis=1)

        accuracy = accuracy_score(real, _predicted)
        hamLoss = hamming_loss(real, _predicted)
        # element wise correctness
        term_wise_accuracy = np.sum(np.logical_not(np.logical_xor(real, _predicted)))/real.size

        macro_precision = precision_score(real, _predicted, average='macro')
        macro_recall = recall_score(real, _predicted, average='macro')
        macro_f1 = f1_score(real, _predicted, average='macro')

        micro_precision = precision_score(real, _predicted, average='micro')
        micro_recall = recall_score(real, _predicted, average='micro')
        micro_f1 = f1_score(real, _predicted, average='micro')

        metricTable = prettytable.PrettyTable()
        metricTable.field_names = ["Metric", "Macro Value", "Micro Value"]
        metricTable.add_row(["Hamming Loss", "{0:.3f}".format(hamLoss), ""])
        metricTable.add_row(["Term Wise Accuracy", "{0:.3f}".format(term_wise_accuracy), ""])

        metricTable.add_row(["Accuracy", "{0:.3f}".format(accuracy), ""])
        metricTable.add_row(["Precision", "{0:.3f}".format(macro_precision), "{0:.3f}".format(micro_precision)])
        metricTable.add_row(["Recall", "{0:.3f}".format(macro_recall), "{0:.3f}".format(micro_recall)])
        metricTable.add_row(["F1-measure", "{0:.3f}".format(macro_f1), "{0:.3f}".format(micro_f1)])

        count_1_as_1 = 0 #TP
        count_1_as_0 = 0 #FN
        count_0_as_1 = 0 #FP
        count_0_as_0 = 0 #TN
        total_real_1s = 0
        total_real_0s = 0
        for i in range(real.shape[0]):
            for j in range(real.shape[1]):
                if real[i,j] == 1:
                    total_real_1s+=1
                    if _predicted[i,j]==1:
                        count_1_as_1 +=1
                    if _predicted[i,j]==0:
                        count_1_as_0 +=1
                if real[i,j] == 0:
                    total_real_0s+=1
                    if _predicted[i,j]==1:
                        count_0_as_1 +=1
                    if _predicted[i,j]==0:
                        count_0_as_0 +=1
        TP = count_1_as_1
        FN = count_1_as_0
        FP = count_0_as_1
        TN = count_0_as_0
        print("count_1_as_1, TP",count_1_as_1)
        print("count_1_as_0, FN",count_1_as_0)
        print("count_0_as_1, FP",count_0_as_1)
        print("count_0_as_0, TN",count_0_as_0)
        print("total_real_1s",total_real_1s)
        print("total_real_0s",total_real_0s)
        # MY_Accuracy = (TP+TN)/(TP+FP+FN+TN)
        # MY_Precision = TP/(TP+FP)
        # MY_Recall = TP/(TP+FN)
        # MY_F1_Score = 2*(MY_Recall * MY_Precision) / (MY_Recall + MY_Precision)
        # print("MY_Accuracy",MY_Accuracy)
        # print("MY_Precision",MY_Precision)
        # print("MY_Recall",MY_Recall)
        # print("MY_F1_Score",MY_F1_Score)
        indepth_metricTable = prettytable.PrettyTable()
        indepth_metricTable.field_names = ["Metric", "Value"]
        indepth_metricTable.add_row(["True Positives, count_1_as_1", "{0:.0f}".format(TP)])
        indepth_metricTable.add_row(["False Negatives, count_1_as_0", "{0:.0f}".format(FN)])
        indepth_metricTable.add_row(["False Positives, count_0_as_1", "{0:.0f}".format(FP)])
        indepth_metricTable.add_row(["True Negatives, count_0_as_0", "{0:.0f}".format(TN)])
        indepth_metricTable.add_row(["Real 1s ", "{0:.0f}".format(total_real_1s)])
        indepth_metricTable.add_row(["Real 0s ", "{0:.0f}".format(total_real_0s)])
        print(metricTable)
        print(indepth_metricTable)
    return predicted


def log_time(key,start_time, end_time, phase,log_dict):
    """log_time [summary]

    Args:
        key ([type]): [name of model]
        start_time ([type]): [start of the phase]
        end_time ([type]): [end of the phase]
        phase ([type]): [phase type. test or train]
        log_dict ([type]): [list of dicts which will store all the files]

    Returns:
        [list of dict]: [appends the data sends the list out]
    """
    log_dict.append({"model":key, "phase":phase,"start_time":start_time ,"end_time":end_time, "total_time":end_time-start_time})
    return log_dict
def time_logger_save(log_dict, filename):
    location= "logs/fit/runtime/"
    pd.DataFrame(log_dict).to_csv(location+filename+".csv")

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

if __name__ == "__main__":
    INPUT_SHAPE =(1,768)
    EM_L_F_UNITS= 768
    EM_L_T_UNITS= 768
    # LEFT CHANNEL
    LSTM_1F_UNITS= 128
    LSTM_1T_UNITS= 128

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
    CONV_4F_KERNEL = 1
    CONV_4T_FILTERS = 12
    CONV_4T_KERNEL = 1

    CONV_3F_FILTERS = 12
    CONV_3F_KERNEL = 1
    CONV_3T_FILTERS = 12
    CONV_3T_KERNEL = 1

    CONV_2F_FILTERS = 12
    CONV_2F_KERNEL = 1
    CONV_2T_FILTERS = 12
    CONV_2T_KERNEL = 1

    LSTM_2_C_L_UNITS = 12

    OUTPUT_DENSE_UNIT =128
    OUTPUT_SIZE =71
    mpstDF= pd.read_csv(r"D:\CodeRepo\Thesis\Thesis\BiLSTM\mpst.csv")
    # LOADING BERT  EMBEDDINGS
    bert_embedding=np.load(r"D:\CodeRepo\Thesis\Thesis\XLNet\xl_embeddings.npz")
    # LOADING XLnet EMBEDDINGS
    xl_embedding=np.load(r"D:\CodeRepo\Thesis\Thesis\XLNet\xl_embeddings.npz")
    # LOADING LABELS Y
    label_values=np.load(r"D:\CodeRepo\Thesis\Thesis\EDA\Y.npz")
    # BERT EMBEDDINGS T1 & T2
    type1_BERT_Embeddings=bert_embedding["t1"]
    type2_BERT_Embeddings=bert_embedding["t2"]
    # XLNet EMBEDDINGS T1 & T2
    type1_XL_Embeddings=xl_embedding["t1"]
    type2_XL_Embeddings=xl_embedding["t2"]
    # LABLES Y
    label_values=label_values["arr_0"]
    # BERT

    # FOR TRAIN
    type1_BERT_Embeddings_Train,type2_BERT_Embeddings_Train,BERT_label_values_Train=get_partition_Embeddings(type1_BERT_Embeddings,type2_BERT_Embeddings,label_values,mpstDF,"train")
    # FOR VALIDATION
    type1_BERT_Embeddings_Val,type2_BERT_Embeddings_Val,BERT_label_values_Val=get_partition_Embeddings(type1_BERT_Embeddings,type2_BERT_Embeddings,label_values,mpstDF,"val")
    # FOR TEST
    type1_BERT_Embeddings_Test,type2_BERT_Embeddings_Test,BERT_label_values_Test=get_partition_Embeddings(type1_BERT_Embeddings,type2_BERT_Embeddings,label_values,mpstDF,"test")


    # XLNET

    # FOR TRAIN
    type1_XL_Embeddings_Train,type2_XL_Embeddings_Train,XLNET_label_values_Train=get_partition_Embeddings(type1_XL_Embeddings,type2_XL_Embeddings,label_values,mpstDF,"train")
    # FOR VALIDATION
    type1_XL_Embeddings_Val,type2_XL_Embeddings_Val,XLNET_label_values_Val=get_partition_Embeddings(type1_XL_Embeddings,type2_XL_Embeddings,label_values,mpstDF,"val")
    # FOR TEST
    type1_XL_Embeddings_Test,type2_XL_Embeddings_Test,XLNET_label_values_Test=get_partition_Embeddings(type1_XL_Embeddings,type2_XL_Embeddings,label_values,mpstDF,"test")

    optimizer_list = ['adam', 'adadelta']
    dataset_X={
        "bert_t1":[
            type1_BERT_Embeddings_Train,
            type1_BERT_Embeddings_Val,
            type1_BERT_Embeddings_Test
            ],
        "bert_t2":[
            type2_BERT_Embeddings_Train,
            type2_BERT_Embeddings_Val,
            type2_BERT_Embeddings_Test
            ],
        "xlnet_t1":[
            type1_XL_Embeddings_Train,
            type1_XL_Embeddings_Val,
            type1_XL_Embeddings_Test
            ],
        "xlnet_t2":[
            type2_XL_Embeddings_Train,
            type2_XL_Embeddings_Val,
            type2_XL_Embeddings_Test
        ]
    }
    dataset_Y ={
        "bert":[
            BERT_label_values_Train,
            BERT_label_values_Val,
            BERT_label_values_Test
            ],
        "xlnet":[
            XLNET_label_values_Train,
            XLNET_label_values_Val,
            XLNET_label_values_Test
            ]
    }

    inp_shape_str = "1_768"

    model_dict ={}
    t_list =[x/10 for x in range(1,10)]
    time_dict=[]

    for ds in dataset_X.keys():
        for opt in optimizer_list:
            print("\nFOR OPTIMIZER: ",opt)
            key= "model_"+inp_shape_str+"_"+opt+"_"+ds
            model_type = ds+"_"+opt
            print("\n###############\nKEY= "+key+"\n#############")
            print("\n###############\nmodel_type= "+model_type+"\n#############")
            model_dict[key] = BalanceNet(INPUT_SHAPE,
                                        EM_L_F_UNITS,
                                        EM_L_T_UNITS,
                                        LSTM_1F_UNITS,
                                        LSTM_1T_UNITS,
                                        CONV_2_FILTER,
                                        CONV_2_KERNEL,
                                        CONV_3_FILTER,
                                        CONV_3_KERNEL,
                                        CONV_5_FILTER,
                                        CONV_5_KERNEL,
                                        CONV_6_FILTER,
                                        CONV_6_KERNEL,
                                        CONV_8_FILTER,
                                        CONV_8_KERNEL,
                                        CONV_4F_FILTERS,
                                        CONV_4F_KERNEL,
                                        CONV_4T_FILTERS,
                                        CONV_4T_KERNEL,
                                        CONV_3F_FILTERS,
                                        CONV_3F_KERNEL,
                                        CONV_3T_FILTERS,
                                        CONV_3T_KERNEL,
                                        CONV_2F_FILTERS,
                                        CONV_2F_KERNEL,
                                        CONV_2T_FILTERS,
                                        CONV_2T_KERNEL,
                                        LSTM_2_C_L_UNITS,
                                        OUTPUT_DENSE_UNIT,
                                        OUTPUT_SIZE,
                                        optimizer_name= opt)
            print("dataset_X["+ds+"][0]:",dataset_X[ds][0].shape)
            print("dataset_X["+ds+"][1]:",dataset_X[ds][1].shape)
            if 'bert' in ds:
                model_dict[key].fitModel(train_x =dataset_X[ds][0], 
                                                    train_y=dataset_Y['bert'][0], 
                                                    val_x= dataset_X[ds][1], 
                                                    val_y = dataset_Y['bert'][1], 
                                                    model_type = model_type, 
                                                    epochs_count = 3 , 
                                                    batch_count = 64, time_dict=time_dict, key=key)
                model_dict[key].predictModel(test_x=dataset_X[ds][2], test_y=dataset_Y['bert'][2], threshold_list=t_list, time_dict=time_dict, key=key)
            if 'xlnet' in ds:
                model_dict[key].fitModel(train_x =dataset_X[ds][0], 
                                                    train_y=dataset_Y['xlnet'][0], 
                                                    val_x= dataset_X[ds][1], 
                                                    val_y = dataset_Y['xlnet'][1], 
                                                    model_type = model_type, 
                                                    epochs_count = 3 , 
                                                    batch_count = 64, time_dict=time_dict, key=key)
                model_dict[key].predictModel(test_x=dataset_X[ds][2], test_y=dataset_Y['xlnet'][2], threshold_list=t_list, time_dict=time_dict, key=key)
    time_logger_save(time_dict,inp_shape_str+"_time")