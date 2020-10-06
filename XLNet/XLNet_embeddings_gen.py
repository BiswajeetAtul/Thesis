# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import collections
import json
import re
import string

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import (preprocess_string, remove_stopwords,
                                          strip_punctuation, strip_tags)
from nltk.corpus import stopwords
from nltk.util import ngrams  # function for making ngrams
from numpy import asarray, save, savez_compressed
from sklearn.feature_extraction.text import CountVectorizer


# %%
import torch
print(torch.cuda.is_available())


# %%
mpstDF= pd.read_csv("mpst.csv")
mpstDF


# %%
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)
    phrase = re.sub(r"couldn't", "could not", phrase)
    phrase = re.sub(r"wouldn't", "would not", phrase)
    phrase = re.sub(r"shouldn't", "should not", phrase)
    phrase = re.sub(r"don't", "do not", phrase)
    phrase = re.sub(r"doesn't", "does not", phrase)
    phrase = re.sub(r"haven't", "have not", phrase)
    phrase = re.sub(r"hasn't", "has not", phrase)
    phrase = re.sub(r"ain't", "not", phrase)
    phrase = re.sub(r"hadn't", "had not", phrase)
    phrase = re.sub(r"didn't", "did not", phrase)
    phrase = re.sub(r"wasn't", "was not", phrase)
    phrase = re.sub(r"aren't", "are not", phrase)
    phrase = re.sub(r"isn't", "is not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# stop_words = stopwords.words('english')


# %%
mpstDF_processsed=mpstDF.copy()
# Type 1: Decontracted Text, The puncutation and stop words are still there
mpstDF_processsed["processed_synopsis_t1"]=mpstDF_processsed["plot_synopsis"].apply(lambda x: decontracted(" ".join(preprocess_string(x, [lambda x: x.lower(), strip_tags]))))
# Type 2 Decontracted Text Stop Words Removed
mpstDF_processsed["processed_synopsis_t2"]=mpstDF_processsed["plot_synopsis"].apply(lambda x: decontracted(" ".join(preprocess_string(x, [lambda x: x.lower(), strip_tags,remove_stopwords]))))

# %% [markdown]
# after the data has been processed, we will truncate the data to 250 and 300 length

# %%
def truncateText(text,size):
    return (" ".join(text.split(" ")[:size]))


# %%
mpstDF_processsed["processed_synopsis_t1_short"]=mpstDF_processsed["processed_synopsis_t1"].apply(lambda x:truncateText(x,250))
# Type 2 Decontracted Text Stop Words Removed
mpstDF_processsed["processed_synopsis_t2_short"]=mpstDF_processsed["processed_synopsis_t2"].apply(lambda x: truncateText(x,300))


# %%
mpstDF_processsed


# %%
# from transformers import XLNetConfig, XLNetModel, XLNetTokenizer
from transformers import AutoModel,AutoConfig,AutoTokenizer


# %%
# xlnConfig= XLNetConfig()
# xlnModel = XLNetModel(xlnConfig)
from transformers import AutoModel,AutoConfig,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
# model = AutoModel.from_config(config)
model = AutoModel.from_pretrained('xlnet-base-cased')


# %%
def get_Encodings(text,tokenizer=tokenizer,model=model, verbose=False):
    if verbose:
        print("Text:")
    print(text[:20] + "....")
    # if the sequence lenght is too big we trim it to 250
    encoded = tokenizer.encode(text)
    
    # if len(encoded)> 248:
    #     encoded = encoded[:248]+[4,3] # adding the encoding for special tokens by default
    if verbose:
        print("Encoded",encoded)
        print("length of Encoded", len(encoded))
    text_encoding_tensor=torch.tensor([encoded])
    if verbose:
        print("text_encoding_tensor:")
        print(text_encoding_tensor)
        print("shape:")
        print(text_encoding_tensor.shape)
    attention_mask_tensor= torch.tensor([[1]*text_encoding_tensor.shape[1]])
    if verbose:
        print("attention_mask_tensor:")
        print(attention_mask_tensor)
        print("shape:")
        print(attention_mask_tensor.shape)

    with torch.no_grad():
        outputs = model(text_encoding_tensor, attention_mask=attention_mask_tensor)
        if verbose:
            print("outputs:")
            print(outputs)
            print("Lenght of outputs",len(outputs))
            print("outputs[0]:")
            print(outputs[0])
            print("outputs[0].shape:")
            print(outputs[0].shape)
            print("outputs[1]:")
            print(outputs[1])
            print("Length Ooutputstput[1]:")
            print(len(outputs[1]))
            print("Sample from Output[1], first hidden layer:")
            print(outputs[1][0])
            print("Sample shape, first hidden layer")
    if verbose:
        print("getting the last tensor for XLNet")
    #print(outputs[0].squeeze()[-1][0:10])
    return outputs[0].squeeze()[-1]


# %%
def getXLNetEmbeddings(df,column,verbose=False):
    embeddings = [np.array(get_Encodings(x,verbose=verbose)) for x in df[column]]
    final_embeddings = np.array(torch.tensor(embeddings))
    return final_embeddings


# %%
# THIS IS NORMAL TEST FUNCTION CREATED AND LATER MNERGED WITH THE getXLNetEmbeddings METHOD
# def getXLNetEmbeddings_testMode(df,column,verbose=False):
#     embeddings = np.array(torch.tensor([np.array(get_Encodings(x,verbose=verbose)) for x in df[column]]))
#     return embeddings

# %% [markdown]
# **Testing verbose mode for one Input**

# %%
sample_text="In May 1980, a Cuban man named Tony Montana (Al Pacino) claims asylum, in Florida, USA, and is in search of the \"American Dream\" after departing Cuba in the Mariel boatlift of 1980. When questioned by three tough-talking INS officials, they notice a tattoo on Tony's left arm of a black heart with a pitchfork through it, which identifies him as a hitman, and detain him in a camp called 'Freedomtown' with other Cubans, including Tony's best friend and former Cuban Army buddy Manolo \"Manny Ray\" Ribiera (Steven Bauer), under the local I-95 expressway while the government evaluates their visa petitions.After 30 days of governmental dithering and camp rumors, Manny receives an offer from the Cuban Mafia which he quickly relays to Tony. If they kill Emilio Rebenga (Roberto Contreras) a former aide to Fidel Castro who is now detained in Freedomtown, they will receive green cards. Tony agrees, and kills Rebenga during a riot at Freedomtown."
test_output=get_Encodings(sample_text)
print(test_output)


# %%


# %% [markdown]
# Now testing the helper functions for input from a dataframe to see if we get the correct desired output for our work

# %%
sampleDF=mpstDF_processsed[["processed_synopsis_t1","processed_synopsis_t2"]].head(3)
sampleDF["processed_synopsis_t1"]=sampleDF["processed_synopsis_t1"].apply(lambda x : x[:1000])
sampleDF["processed_synopsis_t2"]=sampleDF["processed_synopsis_t2"].apply(lambda x : x[:1000])
# display(sampleDF)


# %%
# sample_embeddings=getXLNetEmbeddings_testMode(sampleDF,"processed_synopsis_t1",verbose=True)
sample_embeddings=getXLNetEmbeddings(sampleDF,"processed_synopsis_t1")
print("sample_embeddings.shape",sample_embeddings.shape)
print(sample_embeddings)


# %%
tokenizer.encode("In May 1980, a Cuban man named Tony Montana (Al Pacino) claims asylum, in Florida, USA, and is in search of the \"American Dream\" after departing Cuba in the Mariel boatlift of 1980.")#,"When questioned by three tough-talking INS officials, they notice a tattoo on Tony's left arm of a black heart with a pitchfork through it, which identifies him as a hitman, and detain him in a camp called 'Freedomtown' with other Cubans"])

# %% [markdown]
# HERE: make sure that the output is (3, 768). Where 3 is the number of texts we sent to the model and 768 is the output of embedding length for the model chosen.
# %% [markdown]
# Now We run the model for each text we have in our Dataset 

# %%
print("Starting the Embedding generation for Type 1")


# %%
import time
start_time = time.time()
print(start_time)


# %%
# For Type 1 Embeddings
xlnet_embeddings_t1 = getXLNetEmbeddings(mpstDF_processsed,"processed_synopsis_t1_short")

print("End of the Embedding generation for Type 1")

# %%
print("Shape: ", xlnet_embeddings_t1.shape)
print("XL Embedding for Type 1")
print(xlnet_embeddings_t1)

print("Saving the Embedding generation for Type 1")

# %%
np.savez("xl_embeddings_type1.npz", xlnet_embeddings_t1)
print("Saved!")

# %%
print("Starting the Embedding generation for Type 2")

# For Type 2 Embeddings
xlnet_embeddings_t2 = getXLNetEmbeddings(mpstDF_processsed, "processed_synopsis_t2_short")


# %%
print("Ending the Embedding generation for Type 2")

print("Shape: ", xlnet_embeddings_t2.shape)
print("Embedding")
print(xlnet_embeddings_t2)


# %%
print("--- %s seconds ---" % (time.time() - start_time))

# %%
print("Saving the Embedding generation for Type 2")

np.savez("xl_embeddings_type2.npz", xlnet_embeddings_t2)
print("Saved!")

# %% [markdown]
# Saved the embeddings in different files
# Now saving them in the same file

# %%
print("Saving the Embedding generation for both Types")

np.savez("xl_embeddings.npz", t1=xlnet_embeddings_t1,t2=xlnet_embeddings_t2)
print("Saved!")


# %%
em_check = np.load("xl_embeddings.npz")
print("t1")
print(em_check["t1"])
print(em_check["t1"].shape)
print("t2")
print(em_check["t2"])
print(em_check["t2"].shape)


# %%
print("--- %s seconds ---" % (time.time() - start_time))


# %%



