# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# %%
_identity =np.vectorize(lambda x: x)
_binary_step =np.vectorize(lambda x,t=0: 1 if x>t else 0)
_biploar_step =np.vectorize(lambda x,t=0: 1 if x>t else -1)
_binary_sigmoid=np.vectorize(lambda x: 1. / (1. + np.exp(-x)))
_bipolar_sigmoid=np.vectorize(lambda x: (1. - np.exp(-x))/(1. + np.exp(-x)))
_relu_function=np.vectorize(lambda x: np.max([0, x]))
_relu_leaky = np.vectorize(lambda x: np.max([0.01 * x, x]))
# %%
xlnet_embedding = np.load(r"D:\CodeRepo\Thesis\Thesis\XLNet\xl_embeddings.npz") 
label_values = np.load(r"D:\CodeRepo\Thesis\Thesis\EDA\Y.npz")

type1_XL_Embeddings=xlnet_embedding["t1"]
type2_XL_Embeddings=xlnet_embedding["t2"]
label_values=label_values["arr_0"]
# %%
type1_train_x, type1_test_x, type1_train_y, type1_test_y = train_test_split(type1_XL_Embeddings, label_values, test_size=0.33, random_state=234)
type2_train_x, type2_test_x, type2_train_y, type2_test_y = train_test_split(type2_XL_Embeddings, label_values, test_size=0.33, random_state=230)

# %%
# Running a model with 5000 hidden nodes
elm_5000_t1 = ELM(input_nodes=768, hidden_nodes=5000, output_nodes=71, activation=_bipolar_sigmoid)
elm_5000_t1.fit(type1_train_x, type1_train_y, verbose=False, show_metrics=True)
elm_5000_t1.predict(type1_test_x,type1_test_y, show_metrics=True)


# %%
