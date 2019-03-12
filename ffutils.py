'''
Created on 15 Sep 2017

FanFiction utils to process Harry Potter fanfiction stories

@author: David Vilares
'''

from nltk.tokenize.toktok import ToktokTokenizer  
from nltk.tokenize.stanford import StanfordTokenizer
from multiprocessing.dummy import Pool as ThreadPool 
from collections import OrderedDict
#import nltk.parse.corenlp.CoreNLPTokenizer
from nltk import word_tokenize
from hashedindex import textparser
from bs4 import BeautifulSoup
from random import shuffle
from tqdm import tqdm
import tempfile
import nltk.data
import codecs
import os
import hashedindex
import urllib2
import codecs
import os
import pickle
import time
import random
import tempfile
import uuid
import requests
random.seed(17)

class FanFictionURLCrawler(object):
    
    FF_URL="https://www.fanfiction.net"
    FF_HP_URL = "https://www.fanfiction.net/book/Harry-Potter"
    FF_SORTED_BY="srt"
    FF_LAN="lan"
    FF_STATUS="s"
    FF_RATING="r"
    FF_PAGE="p"
    
    STORY_TITLE = "stitle"
    STORY_DIV_CLASS = "storytext xcontrast_txt nocopy"


    def __init__(self):
        '''
        Constructor
        '''
    
        
    def filter_story_urls(self,lan="en",status="complete",rating="all",
                          rate_limit=2,
                          path_urls="links_to_download.txt",
                          base_url = FF_HP_URL,
                          page=1):
                
            
        f_log  = codecs.open(path_urls,"a") 
        
        map_lan = {"en":"1"}
        
        map_status = {"in-progess":"1",
                      "complete":"2"}
        
        map_rating = {"all":"10"}
        
        there_are_stories = True
        while there_are_stories:
        
            url = base_url+"/?&"+"&".join([self.FF_SORTED_BY+"=1",
                                             self.FF_RATING+"="+map_rating[rating],
                                             self.FF_LAN+"="+map_lan[lan],
                                             self.FF_STATUS+"="+map_status[status],
                                             self.FF_PAGE+"="+str(page)])
            
            response = urllib2.urlopen(url)      
            html = response.read()
            
            print "Page",page
            page+=1

            soup = BeautifulSoup(html,features="html.parser")
            new_stories = soup.findAll("a", class_=self.STORY_TITLE)
            there_are_stories = new_stories != []
            
            if there_are_stories:
                f_log.write("\n".join([self.FF_URL+e["href"] for e in new_stories]))
                f_log.write("\n")
            f_log.flush()
            time.sleep(rate_limit)
        
        f_log.close()
        


class FanFictionHPSpellTokenizer(object):
    
    TOKENIZE_BY_SENTENCE = "sentence"
    TOKENIZE_BY_PARAGRAPH = "paragraph"
    TOKENIZE_AS_TEXT = "text"
    
    DUMMY_SEPARATOR = "DUMMY_SEPARATOR"
    
    def __init__(self,singleword_spells,
                 multiword_spells,tokenize_by="text", #tokenize_by="sentence",
                 punkt_tokenizer='tokenizers/punkt/english.pickle',
                 path_stanford_jar = "/home/david/Descargas/stanford-corenlp-3.8.0.jar"):
  
        self.singleword_spells = singleword_spells
        self.multiword_spells = multiword_spells
        self.multiword_spells_joint = ["_".join(s.split()) for s in multiword_spells]
        self.tokenize_by = tokenize_by
        self.toktok = StanfordTokenizer(path_to_jar=path_stanford_jar)
        self.sent_detector = nltk.data.load(punkt_tokenizer)



    def tokenize(self, path):
        """
        Tokenize one texts at a time is slow
        """                
        with codecs.open(path, encoding="utf-8") as f_fanfiction:
            fanfiction_story = f_fanfiction.read().lower()
            
        if self.tokenize_by == self.TOKENIZE_BY_SENTENCE:
            return self._tokenize_by_sentence(fanfiction_story)
        elif self.tokenize_by == self.TOKENIZE_BY_PARAGRAPH:
            return self._tokenize_by_paragraph(fanfiction_story)
        elif self.tokenize_by == self.TOKENIZE_AS_TEXT:
            return self._tokenize(fanfiction_story)
        else:
            raise NotImplementedError
        
        

    def _tokenize_by_sentence(self, text):
        
        output = []
        sentences = self.sent_detector.tokenize(text.strip()) 

        for s in sentences:
            for mws in self.multiword_spells:
                if mws in s: 
                    s = s.replace(mws,"_".join(mws.split(" ")))
            
        joined_sentences = " "+self.DUMMY_SEPARATOR+" ".join(sentences)        
        new_sentences = " ".join(self.toktok.tokenize(joined_sentences))

        return [s.split(" ") for s in new_sentences.split(self.DUMMY_SEPARATOR)]
        
     
    def _tokenize(self, text):
         
        output = []
        sentences =  " ".join(self.sent_detector.tokenize(text.strip()))
         
        for mws in self.multiword_spells:
            if mws in sentences: 
                sentences = sentences.replace(mws,"_".join(mws.split(" ")))
     
        tokens = self.toktok.tokenize(sentences)  
        output.append(tokens)
            
        return output
    
    
    def is_spell(self,token):
        """
        Tokens must have been obtained after processing the text with the method 
        tokenize()
        """        
        return token in self.multiword_spells_joint or token in self.singleword_spells
    

class SimpleHPSpellsInvertedIndex(object):
    
    NO_SPELL = "NO_SPELL"
    NEW_STORY = "NEW_STORY_SEPARATOR"
    
    CONTEXT_WINDOW_SIZE = "window"
    BATCH_FILE_SPLIT_SYMBOL = " filesplitsymbol "

    def __init__(self,dir_stories, ffhptok, dir_stories_tok, index=None, batch_tok_size=5000):
        
        self.ffhptok = ffhptok   
    
        if index is None:        
            self.index = {}
            ff_paths = [(dir_stories+os.sep+fanfiction, fanfiction)
                            for fanfiction in os.listdir(dir_stories)]             
            ff_content = []     
            ff_ids = []
            
            #############################################################################
            #                               TOKENIZATION
            #       1. We tokenize the fan fiction and store it dir_stories_tok
            #############################################################################
            for ff_batch,(ff_path, ff_name) in enumerate(ff_paths,1):
            
                file_uuid = str(uuid.uuid4())
                with codecs.open("/tmp/"+file_uuid, "w", encoding="utf-8") as f_temp:

                    if ff_batch % batch_tok_size != 0:
                        with codecs.open(ff_path, encoding="utf-8") as f_in:
                            ff_content.append(f_in.read())
                            ff_ids.append(ff_name)
                    else:
                        with codecs.open(ff_path,encoding="utf-8") as f_in:
                            aux = f_in.read()
                            ff_content.append(aux)
                            ff_ids.append(ff_name)
                        
                        f_temp.write(self.BATCH_FILE_SPLIT_SYMBOL.join(ff_content))                            
                        content_tok = self.ffhptok.tokenize(f_temp.name)         
                        #We take below content_tok[0] because in the current tokenization all the document is considered as single sentence
                        #This is done to be able to select snippets of arbritary lengths
                        batch_content = [[ff_content.split(" ")] 
                                         for ff_content in  " ".join(content_tok[0]).split(self.BATCH_FILE_SPLIT_SYMBOL)] 
                        
                        for ffname, ffname_content in zip(ff_ids, batch_content):
                            with codecs.open(dir_stories_tok+os.sep+ffname,"w",encoding="utf-8") as f_out_tok:
                                f_out_tok.write(" ".join(ffname_content[0]))
                        print "Tokenized", ff_batch, "texts"
                        ff_content = []     
                        ff_ids = []

            #We dump the last elements
            with codecs.open("/tmp/"+file_uuid, "w", encoding="utf-8") as f_temp:
                f_temp.write(self.BATCH_FILE_SPLIT_SYMBOL.join(ff_content))                            
                content_tok = self.ffhptok.tokenize(f_temp.name)         
                batch_content = [[ff_content.split(" ")] 
                                 for ff_content in  " ".join(content_tok[0]).split(" filesplitsymbol ")] #len( " ".join(content_tok[0]).split(" filesplitsymbol ") )
              
                for ffname, ffname_content in zip(ff_ids, batch_content):
                    with codecs.open(dir_stories_tok+os.sep+ffname,"w",encoding="utf-8") as f_out_tok:
                        f_out_tok.write(" ".join(ffname_content[0]))   

            ####################################################################################
            #                                  INVERTED INDEX
            # 1. We create an inverted index : spell -> listoccurrences.
            # 2. Each occurrence idenfitied by the ID of the fanfiction plus the index of the token
            # where the spell actually occurs (FFID.0.TOKENIDX). This idenfication strategy depends
            #however on the tokenizer. The user must use the some tokenizer as the one provided with the code
            ####################################################################################
            print "Indexing..."
            for (ff_path,ff_name) in tqdm([(dir_stories_tok+os.sep+f,f) for f in os.listdir(dir_stories_tok)]):
            #for indexing, (ff_path,ff_name) in enumerate([(dir_stories_tok+os.sep+f,f) for f in os.listdir(dir_stories_tok)]):
                    with codecs.open(ff_path, encoding="utf-8") as f:
                        content_tok = [f.read().split(" ")]
                  
                    for sid,s in enumerate(content_tok):
                        spell_in_sentence = False
                        for tid,token in enumerate(s):
                            if self.ffhptok.is_spell(token):
                                if token not in self.index:
                                    self.index[token] = []
                                self.index[token].append(".".join([ff_name, str(sid),str(tid)]))
                                spell_in_sentence = True
        else:
            self.index = index


    def retrieve(self, window, path_stories_tok):
        """
        Gets the content where the spell occurrences occurred
        """
        
        self.window_size = window
        spell_sentences = []
        
        spell_occ = []
        for spell in self.index:
            for fid in self.index[spell]:
                fid_split = fid.split(".")
                fid_name,fid_sid,fid_tid = fid_split[0], fid_split[1], fid_split[2]                             
                spell_occ.append((fid,spell))

        ff_files = set([file_name for file_name in  os.listdir(path_stories_tok)])
        for (fid, spell) in tqdm(spell_occ):
            fid_split = fid.split(".")     
            fid_name,fid_sid,fid_tid = fid_split[0], int(fid_split[1]), int(fid_split[2])
                
            
            with codecs.open(path_stories_tok+os.sep+fid_name, encoding="utf-8") as f:
                content = [f.read().split(" ")]
            sample = self._get_sample_as_window_size(content,int(fid_sid),
                                         int(fid_tid))


            spell_sentences.append((fid, spell.upper(), sample))
            
        return spell_sentences
    


    def _get_sample_as_window_size(self,sentences, sid, tid):
        if tid-self.window_size < 0:
            return " ".join(sentences[sid][0:tid])
        else:
            return " ".join(sentences[sid][(tid-self.window_size):tid])

    
#     def _get_sample_as_a_whole(self, sentences, sid, tid):
#         return " ".join(sentences[sid])
    
#     
#     def _get_sample_as_sentence_size(self, sentences, sid, tid):
#         output = []
#         for ws in range(self.SENTENCE_SIZE,0,-1):
#             if sid-ws >0:
#                 output.append(" ".join(sentences[sid-ws]))
#             
#         output.append(" ".join(sentences[sid][:tid]))
#         
#         return " ".join(output)
#         
                    
                    
    """
    Saves the inverted index with the occurrences of the spells in the FF stories
    """
    def dump(self,file):
        pickle.dump(self,file)


    
    