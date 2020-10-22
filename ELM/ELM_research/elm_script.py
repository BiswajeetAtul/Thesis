

class ELM_MultiLabel:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, activation, bias=True, random_gen="uniform"):
        """
        Args:
            input_nodes ([integer]): Number of Input nodes
            hidden_nodes ([integer]): Number of hidden nodes
            output_nodes ([integer]): Number of output nodes
            activation ([function]): The function which will be used as the activation function in the hidden layer
            bias ([boolean]): Flag to use bias, if True then randomly generate bias @random_gen else bias - 0.
            random_gen (str, optional): The type way in which random weight are generated. Defaults to "uniform".
        """
        self.__input_nodes = input_nodes
        self.__hidden_nodes = hidden_nodes
        self.__output_nodes = output_nodes
        if random_gen == "uniform":
            self.__beta = np.random.uniform(-1., 1.,
                                            size=(self.__hidden_nodes, self.__output_nodes))
            self.__alpha = np.random.uniform(-1., 1.,
                                             size=(self.__input_nodes, self.__hidden_nodes))
            self.__bias = np.random.uniform(size=(self.__hidden_nodes,))
        else:
            self.__beta = np.random.normal(-1., 1.,
                                           size=(self.__hidden_nodes, self.__output_nodes))
            self.__alpha = np.random.normal(-1., 1.,
                                            size=(self.__input_nodes, self.__hidden_nodes))
            self.__bias = np.random.normal(size=(self.__n_hidden_nodes,))
        self.__activation = activation  # Sigmoid Function

    def getInputNodes(self):
        return self.__input_nodes

    def getHiddenNodes(self):
        return self.__hidden_nodes

    def getOutputNodes(self):
        return self.__output_nodes

    def getBetaWeights(self):
        return self.__beta

    def getAlphaWeight(self):
        return self.__alphs

    def getBias(self):
        return self.__bias

    def __get_H_matrix(self, train_x, verbose=False):
        # 1 Propagate data from Input to hidden Layer
        if verbose:
            print("Propagate data from Input to hidden Layer")
        inp = np.dot(train_x, self.__alpha)
        if verbose:
            print(inp)
            print("Adding Biases")
        inp = inp + self.__bias
        if verbose:
            print(inp)
            print("Applyin activation function")
        inp_activation = np.apply_along_axis(self.__activation, 1, inp)
        return inp_activation

    def fit(self, train_x, train_y, verbose=False, show_metrics=True):
        """
        This function calculates the Beta weights or the output weights
        train_x : input matrix
        train_y : output matrix to be predicted or learned upon unipolar

        """
        if verbose:
            print("train_x shape:", train_x.shape)
            print("train_y shape:", train_y.shape)
        inp_activation = self.__get_H_matrix(train_x, verbose)
        # This is the H matrix getting its Moore Penrose Inverse
        if verbose:
            print(inp_activation)
            print("Getting the Generalized Moore Penrose Inverse")
        generalizedInverse = np.linalg.pinv(inp_activation)
        if verbose:
            print(generalizedInverse)
            print("Finding Beta, output weights")
        # Now find output weight matrix Beta
        # convert input Y values according to the threshold using biploar step function
        predicted_bipolar = np.apply_along_axis(_biploar_step, 1, train_y)
        self.__beta = np.dot(generalizedInverse, predicted_bipolar)
        if verbose:
            print("Beta Matrix Weights")
            print(self.__beta)

        if(show_metrics):
            print("Model Metrics, for Training :")
            self.predict(train_x, train_y, verbose, show_metrics)

    def predict(self, test_x, test_y=None, verbose=False, show_metrics=True):
        """
        preditcts the output for the input test data
        call this after calling the fit.
        test_data shape should be (batch_size,768 or input_nodes)
        output_shape will be (batch_size, 71 or output_nodes)

        returns: Predicted Label Matrix
        """
        if verbose:
            print("Predicting outputs")
        inp_activation = self.__get_H_matrix(test_x, verbose)
        output_predicted = np.dot(inp_activation, self.__beta)
        # convert predicted according to the threshold using biploar step function
        predicted_bipolar = np.apply_along_axis(
            _biploar_step, 1, output_predicted)
        predicted_binary = np.apply_along_axis(
            _binary_step, 1, predicted_bipolar)

        if verbose:
            print("predicted output")
            print(output_predicted)
            print("predicted_bipolar")
            print(predicted_bipolar)
            print("predicted_binary")
            print(predicted_binary)
            print("Original Binary")
            print(test_y)

        if(test_y is not None):
            self.__evaluate(test_y, predicted_binary, for_test=False)
        return predicted_binary

    def __evaluate(self, real, predicted, for_test=True):
        """
        real values as 0,1
        predicted values as 0,1
        """
        # Now we find accuracy, precision, recall, Hamming Loss and F1 Measure
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

        print("Metrics @ Literature")
        lit_HamminLosss, lit_accuracy, lit_precision, lit_recall, lit_f1 = self.get_eval_metrics(
            real, predicted)

        return_dict = {"HiddenNodes": self.getHiddenNodes(),
                       "lit_HamminLosss": lit_HamminLosss,
                       "lit_accuracy": lit_accuracy,
                       "lit_precision": lit_precision,
                       "lit_recall": lit_recall,
                       "lit_f1": lit_f1,
                       "sklearn_hamLoss": lit_accuracy,
                       "sklearn_accuracy": accuracy,
                       "sklearn_macro_precision": macro_precision,
                       "sklearn_micro_precision-": lit_accuracy,
                       "sklearn_macro_recall": macro_recall,
                       "sklearn_micro_precision": micro_precision,
                       "sklearn_macro_f1": macro_f1,
                       "sklearn_micro_f1": micro_f1,
                       "term_wise_accuracy": term_wise_accuracy,
                       }

        print("Test Classification Report")
        print(classification_report(real, predicted))

        return return_dict

    def get_eval_metrics(self, real, predicted, verbose=False):
        err_cnt_accuracy = 0
        err_cnt_precision = 0
        err_cnt_recall = 0
        if verbose:
            print(real)
            print(predicted)
        for x in range(real.shape[0]):
            err_and = np.logical_and(real[x], predicted[x])
            err_or = np.logical_or(real[x], predicted[x])
            # Accuracy
            err_cnt_accuracy += (sum(err_and)/sum(err_or))

            # Precision
            if sum(err_and) != 0:
                err_cnt_precision += (sum(err_and) / sum(predicted[x]))
            # Recall
            err_cnt_recall += (sum(err_and) / sum(real[x]))
            if verbose:
                print("Iteration :", x)
                print((sum(err_and)/sum(err_or)))
                print(err_and)
                print(err_or)
                print(err_cnt)

        err_count_hamming = np.zeros((real.shape))

        for i in range(real.shape[0]):
            for j in range(real.shape[1]):
                if real[i, j] != predicted[i, j]:
                    err_count_hamming[1, j] = err_count_hamming[1, j]+1

        sum_err = np.sum(err_count_hamming)
        HammingLoss = sum_err/real.size
        accuracy = err_cnt_accuracy / real.shape[0]
        precision = err_cnt_precision / real.shape[0]
        recall = err_cnt_recall / real.shape[0]
        f1 = 2*((precision*recall)/(precision+recall))
        if verbose:
            print("Final: ")
            print("Hamming Loss: ", HammingLoss)
            print("Accuracy: ", accuracy)
            print("precision: ", precision)
            print("recall: ", recall)
            print("f1: ", f1)

        metricTable = prettytable.PrettyTable()
        metricTable.field_names = ["Metric", "Value"]
        metricTable.add_row([" Literature Hamming Loss",
                             "{0:.3f}".format(HammingLoss)])
        metricTable.add_row(
            ["Literature Accuracy", "{0:.3f}".format(accuracy)])

        metricTable.add_row(
            ["Literature Precision", "{0:.3f}".format(precision)])
        metricTable.add_row(["LiteratureRecall", "{0:.3f}".format(recall)])
        metricTable.add_row(["LiteratureF1-measure", "{0:.3f}".format(f1)])

        print(metricTable)

        return HammingLoss, accuracy, precision, recall, f1


#Validation

for i in list_of_models_hidden_nodes:
    print( "t1_"   +str(i)+"_bipolar_sigmoid_val_start \n")
    time_log["t1_" +str(i)+"_bipolar_sigmoid_val_start"]=time.time()
    predicted, eval_dict=t1_models["t1_"+str(i)+"_bipolar_sigmoid"].predict(type1_BERT_Embeddings_Val,label_values_Val, show_metrics=True)
    time_log["t1_" +str(i)+"_bipolar_sigmoid_val_end"]=time.time()

    add_data_to_metric_list(eval_dict, "_bipolar_sigmoid", "type1_BERT_Embeddings_Val", time_log["t1_" +str(i)+"_bipolar_sigmoid_val_start"], "val", time_log["t1_" +str(i)+"_bipolar_sigmoid_val_end"])


    print( "t1_"   +str(i)+"_relu_leaky_val_start \n")
    time_log["t1_" +str(i)+"_relu_leaky_val_start"]=time.time()
    predicted, eval_dict=t1_models["t1_"+str(i)+"_relu_leaky"].predict(type1_BERT_Embeddings_Val,label_values_Val, show_metrics=True)
    time_log["t1_" +str(i)+"_relu_leaky_val_end"]=time.time()

    add_data_to_metric_list(eval_dict, "_relu_leaky", "type1_BERT_Embeddings_Val", time_log["t1_" +str(i)+"_relu_leaky_val_start"], "val", time_log["t1_" +str(i)+"_relu_leaky_val_end"])


    print( "t1_"   +str(i)+"_biploar_step_val_start \n")
    time_log["t1_" +str(i)+"_biploar_step_val_start"]=time.time()
    predicted, eval_dict=t1_models["t1_"+str(i)+"_biploar_step"].predict(type1_BERT_Embeddings_Val,label_values_Val, show_metrics=True)
    time_log["t1_" +str(i)+"_biploar_step_val_end"]=time.time()

    add_data_to_metric_list(eval_dict, "_biploar_step", "type1_BERT_Embeddings_Val", time_log["t1_" +str(i)+"_biploar_step_val_start"], "val", time_log["t1_" +str(i)+"_biploar_step_val_end"])


    print( "t2_"   +str(i)+"_bipolar_sigmoid_val_start \n")
    time_log["t2_" +str(i)+"_bipolar_sigmoid_val_start"]=time.time()
    predicted, eval_dict=t2_models["t2_"+str(i)+"_bipolar_sigmoid"].predict(type2_BERT_Embeddings_Val,label_values_Val, show_metrics=True)
    time_log["t2_" +str(i)+"_bipolar_sigmoid_val_end"]=time.time()

    add_data_to_metric_list(eval_dict, "_bipolar_sigmoid", "type2_BERT_Embeddings_Val", time_log["t2_" +str(i)+"_bipolar_sigmoid_val_start"], "val", time_log["t2_" +str(i)+"_bipolar_sigmoid_val_end"])


    print( "t2_"   +str(i)+"_relu_leaky_val_start \n")
    time_log["t2_" +str(i)+"_relu_leaky_val_start"]=time.time()
    predicted, eval_dict=t2_models["t2_"+str(i)+"_relu_leaky"].predict(type2_BERT_Embeddings_Val,label_values_Val, show_metrics=True)
    time_log["t2_" +str(i)+"_relu_leaky_val_end"]=time.time()

    add_data_to_metric_list(eval_dict, "_relu_leaky", "type2_BERT_Embeddings_Val", time_log["t2_" +str(i)+"_relu_leaky_val_start"], "val", time_log["t2_" +str(i)+"_relu_leaky_val_end"])

    print( "t2_"   +str(i)+"_biploar_step_val_start \n")
    time_log["t2_" +str(i)+"_biploar_step_val_start"]=time.time()
    predicted, eval_dict=t2_models["t2_"+str(i)+"_biploar_step"].predict(type2_BERT_Embeddings_Val,label_values_Val, show_metrics=True)
    time_log["t2_" +str(i)+"_biploar_step_val_end"]=time.time()

    add_data_to_metric_list(eval_dict, "_biploar_step", "type2_BERT_Embeddings_Val", time_log["t2_" +str(i)+"_biploar_step_val_start"], "val", time_log["t2_" +str(i)+"_biploar_step_val_end"])
