from fanfiction import Scraper
import codecs
import os
import argparse
import requests

my_scraper = Scraper()

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output",
                           help="The directory where each fanfiction story will be writen down (the name of each file will be the ID of the story)")
    argparser.add_argument("--log",
                           help="The path where to store the ids of the URL's that caused some issue and could not be retrieved", type=str)
    argparser.add_argument("--url",
                           help="The text file containing the URLs to crawl")
    argparser.add_argument("--rate_limit",default=2,type=float,
                           help="How fast to crawl fanfiction (in number of seconds)")
    
    try:
        args = argparser.parse_args()
    except:
        argparser.error("Invalid arguments")
        sys.exit(0)
    

path_urls = args.url  
path_output = args.output 
path_log = args.log 
my_scraper.rate_limit = args.rate_limit

f_log = codecs.open(path_log,"a")

with codecs.open(path_urls) as f_urls:
    story_urls = [l.strip() for l in f_urls.readlines()]
    
    if not os.path.exists(path_output):
       os.mkdir(path_output)
       
    for idx_url,url in enumerate(story_urls):
        url_split = url.split("/")
        id_story = int(url_split[-3])
        print "Downloading "+url.strip()+" ("+str(idx_url)+" out of "+str(len(story_urls))+"). Downloading the chapters... ",
        
        try:
            #The new metadata provided by FF does not provide now the number of chapters, so we have to guess. 
            #We follow a conservative approach an predict a very high number.
            #scrape_story() checks inside the real number of chapters
            story_data = my_scraper.scrape_story(id_story, num_chapters=10000)
        
            if 'chapters' in story_data and story_data["chapters"] != {}:
                
                if 1 not in story_data['chapters'] or story_data['chapters'][1] != "":
                    with codecs.open(path_output+os.sep+str(id_story),"w", encoding="utf-8") as f_stories:           
                        for chapter in story_data['chapters']:
                            print chapter,   
                            f_stories.write(story_data['chapters'][chapter].decode("utf-8"))
                else:
                    print "- > fan fiction story not found",
                    f_log.write(url+"\t"+"not_found"+"\n")
            else:
                print "-> chapters not found", #for ",url.strip()
                f_log.write(url+"\t"+"no_chapters"+"\n")
            print           
        except RuntimeError:
            f_log.write(url+"\t"+"RuntimeError"+"\n")
        
        except requests.exceptions.ConnectionError:
            f_log.write(url+"\t"+"requests.exceptions.ConnectionError"+"\n")
            
                   