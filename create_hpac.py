import argparse
import codecs
import pickle
import random
import os
from tqdm import tqdm


def write_split_fullids(split_samples, index_samples, name_split):
    with codecs.open(args.output+os.sep+name_split,"w", encoding="utf-8") as f_split:
        found = False
        for hpac_sample_id in tqdm(split_samples):
            for idx_sample,sample in enumerate(index_samples):
                if hpac_sample_id == sample[0]:
                    found=True
                    f_split.write("\t".join(sample)+"\n")          
                    break
                
            if not found:
                print("Sample not found", hpac_sample)
                exit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--index', 
                        help="Size of the previous snippet")
    parser.add_argument("--dir_stories_tok",
                        help="Path to the directory containing the stories tokenized")
    parser.add_argument("--hpac_train",
                        help="Path to the files containing all HPAC training IDs")
    parser.add_argument("--hpac_dev",
                        help="Path to the files containing all HPAC dev IDs")
    parser.add_argument("--hpac_test",
                        help="Path to the files containing all HPAC dev IDs")
    parser.add_argument("--window_size", type=int,
                        help="Size of the snippet")
    parser.add_argument("--output")
    args = parser.parse_args()



    with codecs.open(args.hpac_train, encoding="utf-8") as f:
        hpac_train = [ l.split("\t")[0] for l in f.readlines()]

    with codecs.open(args.hpac_dev, encoding="utf-8") as f:
        hpac_dev = [ l.split("\t")[0] for l in f.readlines()]

    with codecs.open(args.hpac_test, encoding="utf-8") as f:
        hpac_test = [ l.split("\t")[0] for l in f.readlines()]


    with codecs.open(args.index) as f:
        my_index = pickle.load(f)

    print "Retrieving samples..."
    spell_samples = my_index.retrieve(args.window_size, args.dir_stories_tok)
 
    print "Creating the development set..."
    write_split_fullids(hpac_dev,spell_samples, "hpac_dev_"+str(args.window_size)+".tsv")
    print "Creating the test set..."
    write_split_fullids(hpac_test,spell_samples, "hpac_test_"+str(args.window_size)+".tsv")
    print "Creating the training set..."
    write_split_fullids(hpac_train,spell_samples, "hpac_training_"+str(args.window_size)+".tsv")

 