import argparse
import codecs

"""
It compares the IDs and labels in the generated file with those expected (hpac_XXX_labels.tsv files)
"""

if __name__ == "__main__":
    
    # execute only if run as a script
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--labels",
                           help="Path to the labels files containing ID\tLABEL")
    argparser.add_argument("--input", 
                           help="Path to (generated) input file", type=str)

    try:
        args = argparser.parse_args()
    except:
        argparser.error("Invalid arguments")
        sys.exit(0)
        
    with codecs.open(args.labels, encoding="utf-8") as f_labels:
        empty_samples = {l.split("\t")[0]:l.split("\t")[1].strip() 
                         for l in f_labels.readlines()}
        
    with codecs.open(args.input, encoding="utf-8") as f_in:
        samples = {l.split("\t")[0]:l.split("\t")[1].strip()
                   for l in f_in.readlines()}
    
    errors = False
    generated_not_in_gold = 0  
    labels_mismatch  = 0
    gold_missing = 0
    for emptyidx in empty_samples:
        
        try:
            label = samples[emptyidx]
            if empty_samples[emptyidx] != label:
                print ("Gold sample with identifier [", emptyidx,"] does not match the label in the generated file: ", 
                       empty_samples[emptyidx]," vs ", label)
                errors = True
                labels_mismatch+=1
        except KeyError:
            print ("Gold sample with identifier [", emptyidx,"] is missing in the generated file")
            errors = True
            gold_missing+=1
    for idx in samples:
        try:
            empty_samples[idx]
        except KeyError:
            print ("The generated sample with id [",idx,"] does not appear in the gold file")
            errors = True
            generated_not_in_gold+=1
            
    if not errors:
        print "The file matches 100% the labels file"
    else:
        print 
        print generated_not_in_gold, "generated samples are not in the gold file"
        print labels_mismatch, "samples have a different label in the generated and gold file"
        print gold_missing, "gold samples have not been generated"
        print 
        print "Some samples are missing. Usually the main reasons are:"
        print "1. Fanfiction texts are not available anymore in the website."
        print "2. Wrong tokenization does not allow to correctly build HPAC."
        print "3. A fanfiction text has been modified after this corpus was created and the original ID (FanFictionID.0.TokenID) does not match anymore."
        print 
