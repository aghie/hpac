import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Activation, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
import keras
import codecs
import numpy as np
import model_utils
import models
import argparse
import ConfigParser
import random
import pickle
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--training', metavar='training', type=str, help='Path to the directory containing the fanfiction stories')
parser.add_argument('--test', dest='test', type=str, help='Path to the file containing the spells to take into account')
parser.add_argument("--conf", dest="conf", type=str, help="")
parser.add_argument("--dir", type=str, help="The directory where to store/load the models")
parser.add_argument("--model",dest="model",type=str,help="Options [MAJORITY|MLR,LSTM,CNN,BILSTM,MLP]", default="LG")
parser.add_argument("--predict", dest="predict", action="store_true", default=False)
parser.add_argument("--model_weights", dest="model_weights", default="/tmp/model_weights.hdf5")
parser.add_argument("--model_params", dest="model_params", default="/tmp/model.params")
parser.add_argument("--epochs", dest="epochs", type=int, default=None)
parser.add_argument("--S", dest="S", type=int, default=1)
parser.add_argument("--timesteps", dest="timesteps", default=None)
parser.add_argument("--gpu", dest="gpu")

args = parser.parse_args()


if args.gpu is not None:

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



def save_plot(history, path, title):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
     
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig(path+os.sep+title+'.png')
    plt.close()

    loss_history = np.array(history.history["loss"])
    val_loss_history = np.array(history.history["val_loss"])
    acc_history = np.array(history.history["acc"])
    val_acc_history = np.array(history.history["val_acc"])
    np.savetxt(train_conf["output_dir"]+os.sep+m.name_classifier + "_loss_history.txt", loss_history, delimiter=",")
    np.savetxt(train_conf["output_dir"]+os.sep+m.name_classifier + "_val_loss_history.txt", val_loss_history, delimiter=",")
    np.savetxt(train_conf["output_dir"]+os.sep+m.name_classifier + "_acc_history.txt", acc_history, delimiter=",")
    np.savetxt(train_conf["output_dir"]+os.sep+m.name_classifier + "_val_acc_history.txt", val_acc_history, delimiter=",")



"""
LOADING THE CONFIGURATION FOR THE MODEL
"""
config = ConfigParser.ConfigParser()
config.readfp(open(args.conf))
embeddings =  config.get("Resources",'path_embeddings')
path_spells = config.get("Resources", "path_spells")
batch_size = int(config.get("Settings","batch_size"))


config.set("Settings","output_dir", args.dir)
#Predicting unseen samples
if args.predict:
    
    path_weights = args.model_weights
    path_params = args.model_params
    
    summary_models = []
    for n in range(1,args.S+1):
        size = args.test.replace(".tsv","").split("_")[-1]
        if args.model != models.MAJORITY_NAME:
            args.model_weights = path_weights.replace(".hdf5","_"+str(n)+".hdf5")
            args.model_params = path_params.replace(".params","_"+str(n)+".params")
            
            with codecs.open(args.model_params, 'rb') as paramsfp:
                (words,labels) = pickle.load(paramsfp)
        else:
            with codecs.open(args.model_params, 'rb') as paramsfp:
                (words,labels, majority_class) = pickle.load(paramsfp)

        print ('Initializing model')
        if args.model == models.LG_NAME:
            m = models.LogisticRegressionHP(conf=dict(config.items('MLR')),forms=words, 
                                            labels=labels, options=args)
        elif args.model == models.CNN_NAME:
            
            config.set(args.model,"timesteps", size)
            config.set(args.model,"external_embeddings", embeddings)
            cnn_conf = dict(config.items("CNN"))
            m = models.CNNHP(cnn_conf,vocab=words,labels=labels, options=args)
        elif args.model == models.MLP_NAME:
            config.set(args.model,"timesteps", size)
            mlp_conf = dict(config.items('MLP'))
            m = models.PerceptronHP(mlp_conf,vocab=words,labels=labels, options=args)
        elif args.model == models.LSTM_NAME:
            config.set(args.model,"external_embeddings", embeddings)
            config.set(args.model,"timesteps", size)
            lstm_conf = dict(config.items('LSTM'))
            m = models.LSTMHP(lstm_conf,vocab=words, labels=labels, options=args)
        elif args.model == models.RNN_NAME:
            config.set(args.model,"external_embeddings", embeddings)
            config.set(args.model,"timesteps", size)
            rnn_conf = dict(config.items("RNN"))
            m = models.RNNHP(rnn_conf, vocab=words, labels=labels, options=args)
        elif args.model == models.MAJORITY_NAME:
            m = models.MajorityClassHP(labels=labels, majority_class=majority_class)
        else:
            raise NotImplementedError
    
        with codecs.open(args.test) as f:
            test_lines = f.readlines()

        name_model = m.name_classifier+"_"+size+"_"+str(n)
    
        if args.model != models.MAJORITY_NAME:
            m.model = keras.models.load_model(args.model_weights)
        predictions = m.predict(test_lines)

        results, list_summary, labels = m.evaluate(test_lines, predictions)        
        summary_models.append(list_summary)

        with codecs.open(dict(config.items("Settings"))["output_dir"]+os.sep+name_model+".test_outputs","w") as f:
            f.write("\n".join(labels))
    
        with codecs.open(dict(config.items("Settings"))["output_dir"]+os.sep+name_model+".test_results","w") as f:
            f.write(results)
    
    with codecs.open(dict(config.items("Settings"))["output_dir"]+os.sep+m.name_classifier+"_"+size+".test_summary_results","w") as f:
        
        head = ["Accuracy",
                "P-macro","R-macro","F-macro","P-micro","R-micro","F-micro","P-weighted","R-weighted","F-weighted", "Recall@1", "Recall@2",
                "Recall@5","Recall@10",
                "FP-macro","FR-macro","FF-macro","FP-micro","FR-micro","FF-micro","FP-weighted","FP-weighted","FF-weighted",
                "FRecall@1", "FRecall@2",
                "FRecall@5","FRecall@10",
                "UP-macro","UR-macro","UF-macro","UP-macro","UR-micro","UF-micro","UP-weighted","UR-weighted","UF-weighted",
                "URecall@1","URecall@2",
                "URecall@5","URecall@10",]
        for title, results in zip(head, map(str,[round(value,3)*100 for value in np.mean(summary_models, axis=0)]) ):
            f.write("\t".join([title, results])+"\n")
 

#Training the classifier    
else:


    if args.model == models.LSTM_NAME:
        config.set("LSTM","timesteps", args.timesteps)
    if args.model == models.CNN_NAME:
        config.set("CNN","timesteps", args.timesteps)
    if args.model == models.LG_NAME:
        config.set("MLR","timesteps", args.timesteps)
    if args.model == models.MLP_NAME:
        config.set("MLP","timesteps", args.timesteps)    
    
    config.set(args.model,"external_embeddings", embeddings)
    
    train_conf = dict(config.items("Settings"))
    print 'Loading training data...',
    words, labels  = model_utils.load_data(args.training, path_spells, train=True)
    print "[OK]"
                
    path_weights = args.model_weights
    for n in range(1,args.S+1):
        
        args.model_weights = path_weights.replace(".hdf5","_"+str(n)+".hdf5")
        
        #Instantiating the model
        print ('Initializing model'), args.model
        if args.model == models.LG_NAME:
            m = models.LogisticRegressionHP(conf=dict(config.items('MLR')),forms=words, 
                                            labels=labels, options=args)
        elif args.model == models.CNN_NAME:
            cnn_conf = dict(config.items("CNN"))
            m = models.CNNHP(cnn_conf,vocab=words,labels=labels, options=args)
        elif args.model == models.MLP_NAME:
            mlp_conf = dict(config.items('MLP'))
            m = models.PerceptronHP(mlp_conf,vocab=words,labels=labels, options=args)
        elif args.model == models.LSTM_NAME:
            lstm_conf = dict(config.items('LSTM'))
            m = models.LSTMHP(lstm_conf,vocab=words, labels=labels, options=args)
        elif args.model == models.LSTM_NAME2:
            lstm_conf = dict(config.items('2LSTM'))
            m = models.LSTMHP(lstm_conf,vocab=words, labels=labels, options=args)
#        elif args.model == models.RNN_NAME:
#            rnn_conf = dict(config.items("RNN"))
#            m = models.RNNHP(rnn_conf, vocab=words, labels=labels, options=args)
        elif args.model == models.MAJORITY_NAME:
            m = models.MajorityClassHP(labels=labels)
        else:
            raise NotImplementedError
    
        size = args.training.replace(".tsv","").split("_")[-1]
        
        with codecs.open(args.training) as f:
            train_lines = f.readlines()
    
        with codecs.open(args.test) as f:
            dev_lines = f.readlines()
        
        name_model = m.name_classifier+"_"+size+"_"+str(n)
        history = m.train(train_lines, dev_lines, train_conf, name_model)
        
        if args.model != models.MAJORITY_NAME:
            with open(train_conf["output_dir"]+os.sep+name_model+".params", 'wb') as paramsfp:
                pickle.dump((words, labels), paramsfp)
        else:
            with open(train_conf["output_dir"]+os.sep+name_model+".params", 'wb') as paramsfp:
                pickle.dump((words, labels, m.majority_class), paramsfp)
            
        if history is not None:
            save_plot(history, train_conf["output_dir"], name_model)
            
        predictions = m.predict(dev_lines)

        results, list_summary, _ = m.evaluate(dev_lines, predictions)

        with codecs.open(dict(config.items("Settings"))["output_dir"]+os.sep+name_model+".dev_results","w") as f:
            f.write(results)
            

