from collections import Counter

import argparse
import codecs
import ffutils
import os
import pickle


if __name__ == "__main__":
    
    # execute only if run as a script
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",  help="The directory containing the fanfiction stories", type=str)
    parser.add_argument('--spells',  type=str, help='The file containing the spells to take into account')
    parser.add_argument('--tools', help="The path where to store the index and the tokenizer that were used to create the dataset")
    parser.add_argument("--stanford_jar", dest="stanford_jar", type=str, 
                        help="Path to the Stanford  corenlp jar")
    parser.add_argument("--dir_tok",
                        help="The output directory where to store the tokenized fanfiction stories")
    args = parser.parse_args()

#Reads the keywords that act as keys in the inverted files
with codecs.open(args.spells) as f_spells:
    
    spells = [s.lower().strip() for s in f_spells.readlines()]
    spells_counter = {spell: 0 for spell in spells}
    multiword_spells = [s for s in spells
                        if len(s.split()) >= 2] 
    
    for spell in multiword_spells:
        spells.remove(spell)
        
#Creates a tokenizer specifically intended for Harry Potter FF
ffhptok = ffutils.FanFictionHPSpellTokenizer(spells,multiword_spells,
                                             tokenize_by="text",
                                             path_stanford_jar= args.stanford_jar)

#We dump the tokenizer
with codecs.open(args.tools+os.sep+args.dir.split("/")[-1]+"ff.tokenizer","wb") as f_tok:
    pickle.dump(ffhptok,f_tok)

#Creating the inverted index
print "Creating the index and the tokenized texts... (this will take some minutes)"
index = ffutils.SimpleHPSpellsInvertedIndex(args.dir,
                                            ffhptok,
                                            dir_stories_tok=args.dir_tok#,
                                          #  log_file = codecs.open(args.log_file,"w")
                                            )

with codecs.open(args.tools+os.sep+args.dir.split("/")[-1]+"ff.index","w") as f_index: 
        index.dump(f_index)
