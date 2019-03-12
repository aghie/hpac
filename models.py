from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Activation, Flatten, Input, Permute, Reshape, Lambda, RepeatVector, Masking
from keras.layers import Conv1D, GlobalMaxPooling1D, LSTM, Bidirectional, SimpleRNN
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from prettytable import PrettyTable
from keras.layers import merge
from numpy import argmax, average, argsort
from collections import Counter
#from stop_words import get_stop_words

import keras.backend as K
import os
import keras
import codecs
import numpy as np
import model_utils
import random
import pickle
import sklearn


import matplotlib.pyplot as plt
from sklearn.svm.libsvm import predict

LG_NAME = "MLR"
LSTM_NAME = "LSTM"
LSTM_NAME2= "2LSTM"
BILSTM_NAME = "BILSTM"
CNN_NAME = "CNN"
MLP_NAME = "MLP"
ATT_LSTM_NAME = "ATT_LSTM"
RNN_NAME = "RNN"
MAJORITY_NAME = "MAJORITY"

FILTERS="filters"
KERNEL_SIZE="kernel_size"
NEURONS = "neurons"
MLP_NEURONS="mlp_neurons"
MLP_LAYERS="mlp_layers"
DROPOUT = "dropout"
LAYERS = "layers"
EXTERNAL_EMBEDDINGS = "external_embeddings"
#EXTERNAL_EMBEDDINGS="path_embeddings"
INTERNAL_EMBEDDINGS = "internal_embeddings"
DIM_EMBEDDINGS = "dim_embeddings"
BATCH_SIZE = "batch_size"
EPOCHS = "epochs"
TIMESTEPS = "timesteps"
BIDIRECTIONAL = "bidirectional"
OUTPUT_DIR ="output_dir"


class ModelHP(object):
    
    INDEX_TO_PAD = 0
    UNK_WORD_INDEX = 1
    INIT_REAL_INDEX = 2
    SPECIAL_INDEXES = [INDEX_TO_PAD,UNK_WORD_INDEX]
    
    
    def __init__(self):
        
        self.iforms = None #Re-written in the children classes
        self.ilabels =None #Re-written in the children classes
        self.model = None
    

    def _get_indexes(self,sentence):
        
        input = [0]*(len(self.iforms)+len(self.SPECIAL_INDEXES))
        
        for word in sentence:
            if word in self.iforms:
                index = self.iforms[word]
                input[index]+=1
            
        return np.array(input)
    
    
    
    def generate_data_test(self, lines, batch_size):

        i = 0
        while i < len(lines):
            batch_sample = 0
            x = []
            y = []
                
            while batch_sample < batch_size and i < len(lines):
                
                    
                ls = lines[i].split('\t')   
                y.append(self.ilabels[ls[1]])
                sample = []
                
                for w in ls[2].split():
                    sample.append(w)
                x.append(self._get_indexes(sample))
                
                batch_sample+=1
                i+=1
            
            x = np.array(x)
            y = keras.utils.to_categorical(y, num_classes = len(self.ilabels))
            if batch_size == 1:    
                y.reshape(-1,batch_size,y.shape[1])
            
            yield ([x],[y])
    
    
    def generate_data(self, lines, batch_size):

        i = 0
        while True:
            batch_sample = 0
            x = []
            y = []
                
            while batch_sample < batch_size:
                
                if i >= len(lines): 
                    i=0 #We prepare the indexes for the next iteration
        
                ls = lines[i].split('\t')   
                y.append(self.ilabels[ls[1]])
                sample = []
                
                for w in ls[2].split():
                    sample.append(w)
                x.append(self._get_indexes(sample))
                
                batch_sample+=1
                i+=1
            
            x = np.array(x)
            y = keras.utils.to_categorical(y, num_classes = len(self.ilabels))
            
            yield ([x],[y])
        


    def train(self, training_data, dev_data, train_conf, name_model):
                        
        batch_size = int(train_conf[BATCH_SIZE])
        epochs = int(train_conf[EPOCHS])
        
        save_model_path = train_conf[OUTPUT_DIR]+os.sep+name_model+".hdf5"
        save_callback = keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_acc', verbose=0,
                                                        save_best_only=True, save_weights_only=False)
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=0, mode='auto')
        history = self.model.fit_generator(self.generate_data(training_data,batch_size),
                                           steps_per_epoch = len(training_data)/(batch_size)+1,
                                           epochs = epochs,
                                           validation_data = self.generate_data(dev_data, batch_size),
                                           validation_steps=len(dev_data)/(batch_size)+1,
                                           callbacks = [save_callback,early_stopping_cb],
                                           verbose=1)
        
        return history


    def predict(self, test_data):
        
        batch_size = 128
        output = self.model.predict_generator(generator = self.generate_data_test(test_data,batch_size), 
                                            steps = (len(test_data)/(batch_size))+1 )   
        return output



    def evaluate(self,test_data, test_output):
        
        
        def recall_at_k(preds, gold, k):
            recall = 0.
            assert(len(preds)==len(gold))
            for p,g in zip(preds,gold):
                if g in argsort(-p)[:k]: recall+=1
            
            return recall / len(gold)
            
 

        gold_output = [self.ilabels[sample.split("\t")[1]] for sample in test_data]
        counter_labels = Counter(gold_output)
        average_occ = np.average(counter_labels.values())
        fgold_output = [g for g in gold_output if counter_labels[g] >=  average_occ]
        ugold_output = [g for g in gold_output if counter_labels[g] <  average_occ]
        
        predicted_output = []
        fpredicted_output = []
        upredicted_output = []
        
        test_frequent_output = []
        gold_frequent_output = []
        test_unfrequent_output = []
        gold_unfrequent_output = []
        
        assert(len(gold_output) == len(test_output))
        for j,sample_out in enumerate(test_output):
            
            predicted_output.append(argmax(sample_out))     
            
            if counter_labels[gold_output[j]] >= average_occ:
                test_frequent_output.append(sample_out)
                gold_frequent_output.append(gold_output[j])
                fpredicted_output.append(argmax(sample_out))
            else:
                upredicted_output.append(argmax(sample_out))
                gold_unfrequent_output.append(gold_output[j])
                test_unfrequent_output.append(sample_out)
     
        #Calculating Recall@K with Precision@K and F1-score@K
        recall_at_1 = recall_at_k(test_output, gold_output, 1)
        recall_at_2 = recall_at_k(test_output, gold_output,2)
        recall_at_5 = recall_at_k(test_output, gold_output, 5)
        recall_at_10 = recall_at_k(test_output, gold_output, 10)

        frecall_at_1 = recall_at_k(test_frequent_output, gold_frequent_output, 1)
        frecall_at_2 = recall_at_k(test_frequent_output, gold_frequent_output, 2)
        frecall_at_5 = recall_at_k(test_frequent_output, gold_frequent_output,  5)
        frecall_at_10 = recall_at_k(test_frequent_output, gold_frequent_output, 10)
        
        urecall_at_1 = recall_at_k(test_unfrequent_output, gold_unfrequent_output, 1)
        urecall_at_2 = recall_at_k(test_unfrequent_output, gold_unfrequent_output,2)
        urecall_at_5 = recall_at_k(test_unfrequent_output, gold_unfrequent_output,  5)
        urecall_at_10 = recall_at_k(test_unfrequent_output, gold_unfrequent_output, 10)


        precision, recall, fscore, support = score(gold_output, predicted_output)
        
        accuracy = sklearn.metrics.accuracy_score(gold_output, predicted_output)
        
        fprecision, frecall, ffscore, fsupport = score(fgold_output, fpredicted_output)
        uprecision, urecal, ufscore, usupport = score(ugold_output, upredicted_output)
             
        p_macro, r_macro, f_macro, support_macro = score(gold_output, predicted_output,
                                                         average = "macro")
        p_micro, r_micro, f_micro, support_micro = score(gold_output, predicted_output,
                                                         average = "micro")
        p_weighted, r_weighted, f_weighted, support_weighted = score(gold_output, predicted_output,
                                                         average = "weighted")
        
        fp_macro, fr_macro, ff_macro, fsupport_macro = score(fgold_output, fpredicted_output,
                                                         average = "macro")
        fp_micro, fr_micro, ff_micro, fsupport_micro = score(fgold_output, fpredicted_output,
                                                         average = "micro")
        fp_weighted, fr_weighted, ff_weighted, fsupport_weigthed = score(fgold_output, fpredicted_output,
                                                         average = "weighted")
        
        up_macro, ur_macro, uf_macro, usupport_macro = score(ugold_output, upredicted_output,
                                                         average = "macro")
        up_micro, ur_micro, uf_micro, usupport_micro = score(ugold_output, upredicted_output,
                                                         average = "micro")
        up_weighted, ur_weighted, uf_weighted, usupport_micro = score(ugold_output, upredicted_output,
                                                         average = "weighted")
        
        list_summary = [accuracy, p_macro, r_macro, f_macro, p_micro, r_micro, f_micro, p_weighted, r_weighted, f_weighted,
                        recall_at_1, recall_at_2, recall_at_5, recall_at_10,
        fp_macro, fr_macro, ff_macro, fp_micro, fr_micro, ff_micro, fp_weighted,fr_weighted, ff_weighted,
        frecall_at_1, frecall_at_2, frecall_at_5, frecall_at_10,
        up_macro, ur_macro,uf_macro, up_micro, ur_micro, uf_micro,up_weighted, ur_weighted, uf_weighted,
        urecall_at_1, urecall_at_2 ,urecall_at_5,urecall_at_10 ]
                

        x = PrettyTable()        
        print [self.labelsi[i] for i in range(len(self.labelsi))
                                if i in gold_output or i in predicted_output]
        x.add_column("Label",  [self.labelsi[i] for i in range(len(self.labelsi))
                                if i in gold_output or i in predicted_output])
        x.add_column("Precision", precision)
        x.add_column("Recall", recall)
        x.add_column("F-score", fscore)
        x.add_column("Support", support)
    
        str_output= "Accuracy: "+str(accuracy)+"\n\n"
        str_output+= "F-macro: "+str(f_macro)+"\n"
        str_output+= "P-macro: "+str(p_macro)+"\n"
        str_output+= "R-macro: "+str(r_macro)+"\n\n"
        str_output+= "F-micro: "+str(f_micro)+"\n"
        str_output+= "P-micro: "+str(p_micro)+"\n"
        str_output+= "R-micro: "+str(r_micro)+"\n\n"
        str_output+= "F-weighted: "+str(f_weighted)+"\n"
        str_output+= "P-weighted: "+str(p_weighted)+"\n"
        str_output+= "R-weighted: "+str(r_weighted)+"\n"
        str_output+="Recall@1: "+str(recall_at_1)+"\n"
        str_output+="Recall@2: "+str(recall_at_2)+"\n"
        str_output+="Recall@5: "+str(recall_at_5)+"\n"
        str_output+="Recall@10: "+str(recall_at_10)+"\n\n"
        
        
        
        str_output+= "-------ONLY FREQUENT SPELLS-------\n\n"
        
        str_output+= "F-macro (>="+str(average_occ)+"): "+str(ff_macro)+"\n"
        str_output+= "P-macro (>="+str(average_occ)+"): "+str(fp_macro)+"\n"
        str_output+= "R-macro (>="+str(average_occ)+"): "+str(fr_macro)+"\n\n"
        str_output+= "F-micro (>="+str(average_occ)+"): "+str(ff_micro)+"\n"
        str_output+= "P-micro (>="+str(average_occ)+"): "+str(fp_micro)+"\n"
        str_output+= "R-micro (>="+str(average_occ)+"): "+str(fr_micro)+"\n\n"
        str_output+= "F-weighted (>="+str(average_occ)+"): "+str(ff_weighted)+"\n"
        str_output+= "P-weighted (>="+str(average_occ)+"): "+str(fp_weighted)+"\n"
        str_output+= "R-weighted (>="+str(average_occ)+"): "+str(fr_weighted)+"\n\n"
        str_output+="Recall@1: "+str(frecall_at_1)+"\n"
        str_output+="Recall@2: "+str(frecall_at_2)+"\n"
        str_output+="Recall@5: "+str(frecall_at_5)+"\n"
        str_output+="Recall@10: "+str(frecall_at_10)+"\n"
        
        
        str_output+= "-------ONLY UNFREQUENT SPELLS-------\n\n"
        
        str_output+= "F-macro: (<"+str(average_occ)+"): "+str(uf_macro)+"\n"
        str_output+= "P-macro: (<"+str(average_occ)+"): "+str(up_macro)+"\n"
        str_output+= "R-macro: (<"+str(average_occ)+"): "+str(ur_macro)+"\n\n"
        str_output+= "F-micro: (<"+str(average_occ)+"): "+str(uf_micro)+"\n"
        str_output+= "P-micro: (<"+str(average_occ)+"): "+str(up_micro)+"\n"
        str_output+= "R-micro: (<"+str(average_occ)+"): "+str(ur_micro)+"\n\n"
        str_output+= "F-weighted (>="+str(average_occ)+"): "+str(uf_weighted)+"\n"
        str_output+= "P-weighted (>="+str(average_occ)+"): "+str(up_weighted)+"\n"
        str_output+= "R-weighted (>="+str(average_occ)+"): "+str(ur_weighted)+"\n\n"
        str_output+="Recall@1: "+str(urecall_at_1)+"\n"
        str_output+="Recall@2: "+str(urecall_at_2)+"\n"
        str_output+="Recall@5: "+str(urecall_at_5)+"\n"
        str_output+="Recall@10: "+str(urecall_at_10)+"\n\n"
        
        str_output+= "-------SUMMARY---------\n\n"
        
        str_output+=str(x)
        
        #print str(x)

        labels = [self.labelsi[l] for l in predicted_output]
        return str_output, list_summary, labels

        
        
        
class  MajorityClassHP(ModelHP):
    
    def __init__(self,labels, majority_class=None):
        
        self.name_classifier = "HP_MAJORITY"
        self.majority_class = majority_class
        
        self.ilabels ={l:i for i,l in enumerate(sorted(labels))}
        self.labelsi = {self.ilabels[l]: l for l in self.ilabels}
        
        print self.ilabels, len(self.ilabels)
        print self.labelsi, len(self.labelsi)
        
        
    def train(self, training_data, dev_data, train_conf, name_model):
    
        labels = []
        for sample in training_data:
            sample_split = sample.split("\t")
            label,text = sample_split[1], sample_split[2]
            labels.append(label)
        counter = Counter(labels)
        
        self.majority_class = counter.most_common(1)[0][0]
    
        
        return None
    
    def predict(self, test_data):
        
        return [self.ilabels[self.majority_class]]*len(test_data)
    

    def evaluate(self,test_data, predicted_output):
        
        
        
        def recall_at_k(preds, gold, k):
            recall = 0.

            for p,g in zip(preds,gold):
                if g == p: recall+=1
            
            return recall / len(gold)
             
        
         
        gold_output = [self.ilabels[sample.split("\t")[1]] 
                       for sample in test_data]
        
        counter_labels = Counter(gold_output)
        
        average_occ = np.average(counter_labels.values())
        
        fgold_output = [g for g in gold_output if counter_labels[g] >=  average_occ]
        ugold_output = [g for g in gold_output if counter_labels[g] <  average_occ]
        
        fpredicted_output, upredicted_output = [],[]

        test_frequent_output = []
        gold_frequent_output = []
        test_unfrequent_output = []
        gold_unfrequent_output = []
        
        
        assert(len(gold_output) == len(predicted_output))
        
        for j,sample_out in enumerate(predicted_output):
            
            if counter_labels[gold_output[j]] >= average_occ:
                fpredicted_output.append(sample_out)
                test_frequent_output.append(sample_out)
                gold_frequent_output.append(gold_output[j])
            else:
                upredicted_output.append(sample_out)
                gold_unfrequent_output.append(gold_output[j])
                test_unfrequent_output.append(sample_out)
        
        #Calculating Recall@K with Precision@K and F1-score@K
        recall_at_1 = recall_at_k(predicted_output, gold_output, 1)
        recall_at_2 = recall_at_k(predicted_output, gold_output,2)
        recall_at_5 = recall_at_k(predicted_output, gold_output, 5)
        recall_at_10 = recall_at_k(predicted_output, gold_output, 10)

        frecall_at_1 = recall_at_k(test_frequent_output, gold_frequent_output, 1)
        frecall_at_2 = recall_at_k(test_frequent_output, gold_frequent_output, 2)
        frecall_at_5 = recall_at_k(test_frequent_output, gold_frequent_output,  5)
        frecall_at_10 = recall_at_k(test_frequent_output, gold_frequent_output, 10)
        
        urecall_at_1 = recall_at_k(test_unfrequent_output, gold_unfrequent_output, 1)
        urecall_at_2 = recall_at_k(test_unfrequent_output, gold_unfrequent_output,2)
        urecall_at_5 = recall_at_k(test_unfrequent_output, gold_unfrequent_output,  5)
        urecall_at_10 = recall_at_k(test_unfrequent_output, gold_unfrequent_output, 10)
        
        
        
        
                 
        precision, recall, fscore, support = score(gold_output, predicted_output)
        
        accuracy = sklearn.metrics.accuracy_score(gold_output, predicted_output)
        
        fprecision, frecall, ffscore, fsupport = score(fgold_output, fpredicted_output)
        uprecision, urecal, ufscore, usupport = score(ugold_output, upredicted_output)
             
        p_macro, r_macro, f_macro, support_macro = score(gold_output, predicted_output,
                                                         average = "macro")
        p_micro, r_micro, f_micro, support_micro = score(gold_output, predicted_output,
                                                         average = "micro")
        p_weighted, r_weighted, f_weighted, support_weighted = score(gold_output, predicted_output,
                                                         average = "weighted")
        
        fp_macro, fr_macro, ff_macro, fsupport_macro = score(fgold_output, fpredicted_output,
                                                         average = "macro")
        fp_micro, fr_micro, ff_micro, fsupport_micro = score(fgold_output, fpredicted_output,
                                                         average = "micro")
        fp_weighted, fr_weighted, ff_weighted, fsupport_weigthed = score(fgold_output, fpredicted_output,
                                                         average = "weighted")
        
        up_macro, ur_macro, uf_macro, usupport_macro = score(ugold_output, upredicted_output,
                                                         average = "macro")
        up_micro, ur_micro, uf_micro, usupport_micro = score(ugold_output, upredicted_output,
                                                         average = "micro")
        up_weighted, ur_weighted, uf_weighted, usupport_micro = score(ugold_output, upredicted_output,
                                                         average = "weighted")
        
        
        list_summary = [accuracy, p_macro, r_macro, f_macro, p_micro, r_micro, f_micro, p_weighted, r_weighted, f_weighted,
                        recall_at_1, recall_at_2, recall_at_5, recall_at_10,
        fp_macro, fr_macro, ff_macro, fp_micro, fr_micro, ff_micro, fp_weighted,fr_weighted, ff_weighted,
        frecall_at_1, frecall_at_2, frecall_at_5, frecall_at_10,
        up_macro, ur_macro,uf_macro, up_micro, ur_micro, uf_micro,up_weighted, ur_weighted, uf_weighted,
        urecall_at_1, urecall_at_2, urecall_at_5, urecall_at_10]


        x = PrettyTable()        
        print [self.labelsi[i] for i in range(len(self.labelsi))
                                if i in gold_output or i in predicted_output]
        x.add_column("Label",  [self.labelsi[i] for i in range(len(self.labelsi))
                                if i in gold_output or i in predicted_output])
        x.add_column("Precision", precision)
        x.add_column("Recall", recall)
        x.add_column("F-score", fscore)
        x.add_column("Support", support)
    
        str_output= "Accuracy: "+str(accuracy)+"\n\n"
        str_output+= "F-macro: "+str(f_macro)+"\n"
        str_output+= "P-macro: "+str(p_macro)+"\n"
        str_output+= "R-macro: "+str(r_macro)+"\n\n"
        str_output+= "F-micro: "+str(f_micro)+"\n"
        str_output+= "P-micro: "+str(p_micro)+"\n"
        str_output+= "R-micro: "+str(r_micro)+"\n\n"
        str_output+= "F-weighted: "+str(f_weighted)+"\n"
        str_output+= "P-weighted: "+str(p_weighted)+"\n"
        str_output+= "R-weighted: "+str(r_weighted)+"\n"
        str_output+="Recall@1: "+str(recall_at_1)+"\n"
        str_output+="Recall@2: "+str(recall_at_2)+"\n"
        str_output+="Recall@5: "+str(recall_at_5)+"\n"
        str_output+="Recall@10: "+str(recall_at_10)+"\n\n"
        
        str_output+= "-------ONLY FREQUENT SPELLS-------\n\n"
        
        str_output+= "F-macro (>="+str(average_occ)+"): "+str(ff_macro)+"\n"
        str_output+= "P-macro (>="+str(average_occ)+"): "+str(fp_macro)+"\n"
        str_output+= "R-macro (>="+str(average_occ)+"): "+str(fr_macro)+"\n\n"
        str_output+= "F-micro (>="+str(average_occ)+"): "+str(ff_micro)+"\n"
        str_output+= "P-micro (>="+str(average_occ)+"): "+str(fp_micro)+"\n"
        str_output+= "R-micro (>="+str(average_occ)+"): "+str(fr_micro)+"\n\n"
        str_output+= "F-weighted (>="+str(average_occ)+"): "+str(ff_weighted)+"\n"
        str_output+= "P-weighted (>="+str(average_occ)+"): "+str(fp_weighted)+"\n"
        str_output+= "R-weighted (>="+str(average_occ)+"): "+str(fr_weighted)+"\n"
        str_output+="Recall@1: "+str(frecall_at_1)+"\n"
        str_output+="Recall@2: "+str(frecall_at_2)+"\n"
        str_output+="Recall@5: "+str(frecall_at_5)+"\n"
        str_output+="Recall@10: "+str(frecall_at_10)+"\n"
        
        str_output+= "-------ONLY UNFREQUENT SPELLS-------\n\n"
        
        str_output+= "F-macro: (<"+str(average_occ)+"): "+str(uf_macro)+"\n"
        str_output+= "P-macro: (<"+str(average_occ)+"): "+str(up_macro)+"\n"
        str_output+= "R-macro: (<"+str(average_occ)+"): "+str(ur_macro)+"\n\n"
        str_output+= "F-micro: (<"+str(average_occ)+"): "+str(uf_micro)+"\n"
        str_output+= "P-micro: (<"+str(average_occ)+"): "+str(up_micro)+"\n"
        str_output+= "R-micro: (<"+str(average_occ)+"): "+str(ur_micro)+"\n\n"
        str_output+= "F-weighted (>="+str(average_occ)+"): "+str(uf_weighted)+"\n"
        str_output+= "P-weighted (>="+str(average_occ)+"): "+str(up_weighted)+"\n"
        str_output+= "R-weighted (>="+str(average_occ)+"): "+str(ur_weighted)+"\n"
        str_output+="Recall@1: "+str(urecall_at_1)+"\n"
        str_output+="Recall@2: "+str(urecall_at_2)+"\n"
        str_output+="Recall@5: "+str(urecall_at_5)+"\n"
        str_output+="Recall@10: "+str(urecall_at_10)+"\n\n"
        
        str_output+= "-------SUMMARY---------\n\n"
        
        str_output+=str(x)
        
        print str(x)
        labels = [self.labelsi[l] for l in predicted_output]
     #   print [self.labelsi[lid] for lid in predicted_output]

        return str_output, list_summary, labels

        

class LogisticRegressionHP(ModelHP):
    """
    A logistic regression implemented with Keras
    """

    def __init__(self,conf,forms, labels, options):
        
        self.name_classifier = "HP_MLR"
        self.iforms = {w:self.INIT_REAL_INDEX+i for i,w in enumerate(sorted(forms))}
        self.conf = conf
        self.ilabels ={l:i for i,l in enumerate(sorted(labels))}
        
        self.labelsi = {self.ilabels[l]: l for l in self.ilabels}
        
        self.n_labels = len(self.ilabels)
        
        model = Sequential() 
        model.add(Dense(self.n_labels, input_dim=len(self.iforms)+len(self.SPECIAL_INDEXES), activation='softmax')) 
        
        model.compile(loss='categorical_crossentropy',
        optimizer="adam",#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),#'sgd',
        metrics=['accuracy'])
    
        self.model = model
        
        self.options = options




class PerceptronHP(ModelHP):

    
    def __init__(self,conf, vocab,labels, options):
        
        self.name_classifier = "HP_MLP"
        self.conf = conf
        self.ilabels ={l:i for i,l in enumerate(sorted(labels))}
        self.n_labels = len(self.ilabels)
        self.labelsi = {self.ilabels[l]: l for l in self.ilabels}
        self.iforms = {w:self.INIT_REAL_INDEX+i for i,w in enumerate(sorted(vocab))}
        input_iw = Input(shape=(len(self.iforms)+len(self.SPECIAL_INDEXES),), name="input", dtype='float32')
        
        x = input_iw
        for l in range(0,int(self.conf[LAYERS])):
            x = Dense(int(self.conf[NEURONS]))(x)
            x = Dropout(float(self.conf[DROPOUT]))(x)          
            x = Activation('relu')(x)
    
        x = Dense(self.n_labels)(x) 
        output = Activation('softmax', name='output')(x)
        
        model = Model(inputs = [input_iw], outputs = [output])
              
        model.compile(loss="categorical_crossentropy",
                    optimizer="adam",#keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=1e-6, nesterov=False),
                    metrics=['accuracy'])           
        
        self.model = model

        self.options = options




class SequenceModelHP(ModelHP):


    def _get_indexes(self,sentence):
        
        input = []
        input_ext = []
        for word in sentence:
      
            index = self.UNK_WORD_INDEX
            
            if word in self.iforms and self.vocab[word] > 1:
                index = self.iforms[word] 
            elif word.lower() in self.iforms and self.vocab[word] > 1:
                index = self.iforms[word.lower()]
            input.append(index)
            
            index = self.UNK_WORD_INDEX
            
            if word in self.ieforms:
                index = self.ieforms[word]
            elif word.lower() in self.ieforms:
                index = self.ieforms[word.lower()]
                
            input_ext.append(index)
            
        return np.array(input), np.array(input_ext)


    def generate_data_test(self, lines, batch_size):

    
        i = 0
        while i < len(lines):
        #while True:
            batch_sample = 0
            x = []
            x_ext = []
            y = []

            while batch_sample < batch_size and i<len(lines):
                
                ls = lines[i].split('\t')   
                
                y.append(self.ilabels[ls[1]])
        
                sample = []
                sample =  ls[2].split()[-self.sequence_length:]

                x1,x2 = self._get_indexes(sample)
                x.append(x1)
                x_ext.append(x2)
                
                batch_sample+=1
                i+=1
            
            
            x = keras.preprocessing.sequence.pad_sequences(x, maxlen=self.sequence_length, dtype='int32',
                                                           truncating='pre', value=0.)
            x_ext = keras.preprocessing.sequence.pad_sequences(x, maxlen=self.sequence_length, dtype='int32',
                                                           truncating='pre', value=0.)

            x = np.array(x)
            x_ext = np.array(x_ext)
            y = keras.utils.to_categorical(y, num_classes = len(self.ilabels))    
            if batch_size == 1:    
                y.reshape(-1,batch_size,y.shape[1])
            
            yield ([x,x_ext],[y])



    def generate_data(self, lines, batch_size):

    
        i = 0
        while True:
            batch_sample = 0
            x = []
            x_ext = []
            y = []

            while batch_sample < batch_size:
                
                if i >= len(lines): 
                    i=0 #We prepare the indexes for the next iteration
        
                ls = lines[i].split('\t')   
                
                y.append(self.ilabels[ls[1]])
        
                sample = []
                sample =  ls[2].split()[-self.sequence_length:]

                x1,x2 = self._get_indexes(sample)
                x.append(x1)
                x_ext.append(x2)               
                batch_sample+=1
                i+=1
            
            x = keras.preprocessing.sequence.pad_sequences(x, maxlen=self.sequence_length, dtype='int32',
                                                           truncating='pre', value=0.)
            x_ext = keras.preprocessing.sequence.pad_sequences(x, maxlen=self.sequence_length, dtype='int32',
                                                           truncating='pre', value=0.)
                  
            x = np.array(x)
            x_ext = np.array(x_ext)
            y = keras.utils.to_categorical(y, num_classes = len(self.ilabels))        
            y.reshape(-1,batch_size,y.shape[1])
            
            yield ([x,x_ext],[y])


class RNNHP(SequenceModelHP):

    def __init__(self, conf, vocab, labels, options):

        self.vocab = vocab
        self.name_classifier = "HP_RNN"
        self.conf = conf
        self.ilabels ={l:i for i,l in enumerate(sorted(labels))}
        self.n_labels = len(self.ilabels)
        self.labelsi = {self.ilabels[l]: l for l in self.ilabels}
        
        self.iforms = {w:self.INIT_REAL_INDEX+i for i,w in enumerate(sorted(vocab))}
        self.dims = int(self.conf[DIM_EMBEDDINGS])
        self.w_lookup = np.zeros(shape=(len(vocab) + len(self.SPECIAL_INDEXES),self.dims))       
        self.ieforms, self.ew_lookup, self.edims = model_utils._read_embedding_file(self.conf[EXTERNAL_EMBEDDINGS])
        
        self.iforms_reverse = {self.iforms[w]:w for w in self.iforms}
        self.ieforms_reverse = {self.ieforms[w]:w for w in self.ieforms}
        
        self.sequence_length = int(self.conf[TIMESTEPS])

        input = Input(shape=(self.sequence_length,), dtype='float32')
        input_ext = Input(shape=(self.sequence_length,), dtype='float32')

        embedding_layer = Embedding(self.w_lookup.shape[0],
                                    self.dims,
                                    embeddings_initializer="glorot_uniform",
                                    input_length=self.sequence_length,
                                    name = "e_IW",
                                    trainable=True)(input)

        e_embedding_layer = Embedding(self.ew_lookup.shape[0],
                                    self.edims,
                                    weights=[self.ew_lookup],
                                    input_length=self.sequence_length,
                                    name = "e_EW",
                                    trainable=True)(input_ext)
        
        x = keras.layers.concatenate([embedding_layer, e_embedding_layer], axis=-1)
        
        dr = float(self.conf[DROPOUT])
        
        for l in range(0,int(self.conf[LAYERS])):
        
            if l == len(range(0, int(self.conf[LAYERS]))):
                sequences = True
            else:
                sequences = False      
            x = SimpleRNN(int(self.conf[NEURONS]), return_sequences=sequences, dropout= dr)(x)

        for l in range(0, int(self.conf[MLP_LAYERS])):
            x = Dense(int(self.conf[MLP_NEURONS]))(x)
            x = Dropout(float(self.conf[DROPOUT]))(x)  
            x = Activation('relu')(x)
            
        preds = Dense(self.n_labels, activation='softmax')(x)
        model = Model(inputs = [input,input_ext], outputs=[preds])        
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.model = model
        self.options = options




class LSTMHP(SequenceModelHP):
    
    def __init__(self, conf, vocab,labels, options):
        
        self.vocab = vocab
        self.name_classifier = "HP_LSTM"
        self.conf = conf
        self.ilabels ={l:i for i,l in enumerate(sorted(labels))}
        self.n_labels = len(self.ilabels)
        self.labelsi = {self.ilabels[l]: l for l in self.ilabels}
        
        self.iforms = {w:self.INIT_REAL_INDEX+i for i,w in enumerate(sorted(vocab))}
        self.dims = int(self.conf[DIM_EMBEDDINGS])
        self.w_lookup = np.zeros(shape=(len(vocab) + len(self.SPECIAL_INDEXES),self.dims))  
             
        self.ieforms, self.ew_lookup, self.edims = model_utils._read_embedding_file(self.conf[EXTERNAL_EMBEDDINGS])
        
        self.iforms_reverse = {self.iforms[w]:w for w in self.iforms}
        self.ieforms_reverse = {self.ieforms[w]:w for w in self.ieforms}
        self.sequence_length = int(self.conf[TIMESTEPS])

        input = Input(shape=(self.sequence_length,), dtype='float32')
        input_ext = Input(shape=(self.sequence_length,), dtype='float32')

        embedding_layer = Embedding(self.w_lookup.shape[0],
                                    self.dims,
                                    embeddings_initializer="glorot_uniform",
                                    input_length=self.sequence_length,
                                    name = "e_IW",
                                    trainable=True)(input)

        e_embedding_layer = Embedding(self.ew_lookup.shape[0],
                                    self.edims,
                                    weights=[self.ew_lookup],
                                    input_length=self.sequence_length,
                                    name = "e_EW",
                                    trainable=True)(input_ext)
        
        
        x = keras.layers.concatenate([embedding_layer, e_embedding_layer], axis=-1)
        bidirectional = self.conf[BIDIRECTIONAL].lower() == "true"
        dr = float(self.conf[DROPOUT])
        
        for l in range(1,int(self.conf[LAYERS])+1):
    
            if l < int(self.conf[LAYERS]):
                sequences = True
            else:
                sequences = False
        
            if bidirectional:
                x = Bidirectional(LSTM(int(self.conf[NEURONS]), dropout=dr, recurrent_dropout=dr,
                                       return_sequences=sequences))(x)
            else:
             
                x = LSTM(int(self.conf[NEURONS]), dropout=dr, #recurrent_dropout=dr,
                         return_sequences=sequences)(x)
                
        for l in range(0, int(self.conf[MLP_LAYERS])):
            x = Dense(int(self.conf[MLP_NEURONS]))(x)
            x = Dropout(float(self.conf[DROPOUT]))(x)  
            x = Activation('relu')(x)
            
        preds = Dense(self.n_labels, activation='softmax')(x)
        model = Model(inputs = [input,input_ext], outputs=[preds])
        model.compile(loss='categorical_crossentropy',
                      optimizer="adam", #keras.optimizers.Adam(lr=1e-3, decay=0),#keras.optimizers.SGD(lr=0.01, momentum=0.7, decay=0.0, nesterov=False),
                      metrics=['accuracy'])

        model.summary()

        self.model = model        
        self.options = options

    

class CNNHP(SequenceModelHP):
    
    def __init__(self, conf,vocab,labels, options):

        self.vocab = vocab
        self.name_classifier = "HP_CNN"
        self.conf = conf
        
        self.ilabels ={l:i for i,l in enumerate(sorted(labels))}
        self.n_labels = len(self.ilabels)
        self.labelsi = {self.ilabels[l]: l for l in self.ilabels}
        
        self.iforms = {w:self.INIT_REAL_INDEX+i for i,w in enumerate(sorted(vocab))}
        self.dims = int(self.conf[DIM_EMBEDDINGS])
        self.w_lookup = np.zeros(shape=(len(vocab) + len(self.SPECIAL_INDEXES),self.dims))       
        
        self.ieforms, self.ew_lookup, self.edims = model_utils._read_embedding_file(self.conf[EXTERNAL_EMBEDDINGS])
        
        self.iforms_reverse = {self.iforms[w]:w for w in self.iforms}
        self.ieforms_reverse = {self.ieforms[w]:w for w in self.ieforms}
        
        self.sequence_length = int(self.conf[TIMESTEPS])

        input = Input(shape=(self.sequence_length,), dtype='float32')
        input_ext = Input(shape=(self.sequence_length,), dtype='float32')

        embedding_layer = Embedding(self.w_lookup.shape[0],
                                    self.dims,
                                    embeddings_initializer="glorot_uniform",
                                    #weights=[self.w_lookup],
                                    input_length=self.sequence_length,
                                    name = "e_IW",
                                    trainable=True)(input)
        
        e_embedding_layer = Embedding(self.ew_lookup.shape[0],
                                    self.edims,
                                    weights=[self.ew_lookup],
                                    input_length=self.sequence_length,
                                    name = "e_EW",
                                    trainable=True)(input_ext)
        
        x = keras.layers.concatenate([embedding_layer, e_embedding_layer], axis=-1)
        

        dr = float(self.conf[DROPOUT])
        filters = int(self.conf[FILTERS])
        kernel_size =int(self.conf[KERNEL_SIZE])

        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        
        for l in range(0, int(self.conf[LAYERS])):
            
            x =Conv1D(filters,
                      kernel_size,
                      padding='valid',
                      activation='relu',
                      strides=1)(x)

        x = GlobalMaxPooling1D()(x)
        
        
        for l in range(0, int(self.conf[MLP_LAYERS])):
            x = Dense(int(self.conf[MLP_NEURONS]))(x)
            x = Dropout(float(self.conf[DROPOUT]))(x)  
            x = Activation('relu')(x)
            
        preds = Dense(self.n_labels, activation='softmax')(x)
        model = Model(inputs = [input, input_ext], outputs=[preds])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()

        self.model = model




