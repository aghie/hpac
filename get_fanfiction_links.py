'''
Created on 15 Sep 2017

@author: david.vilares
'''

from ffutils import *
import argparse


if __name__ == "__main__":
    
    # execute only if run as a script
    argparser = argparse.ArgumentParser(description='Extract statistics for the HPARC corpyss')
    argparser.add_argument("--base_url",
                           default= "https://www.fanfiction.net/book/Harry-Potter/",
                           help="The URL from where to download fanfiction")
    argparser.add_argument("--lang", default="en", 
                           help="Download stories written in a given language", type=str)
    argparser.add_argument("--status",
                           help="Download fanfiction with a certain status",
                           default="complete")
    argparser.add_argument("--rating", default="all",
                           help="Download fanfiction with a certain rating")
    argparser.add_argument("--page", type=int,
                           help="Save links from this page onwards")
    argparser.add_argument("--rate_limit",default=2.0,type=float,
                           help="Float that indicates the speed you can read stories in fanfiction")
    argparser.add_argument("--output",
                           help="The path where to write the URLS")
    
    try:
        args = argparser.parse_args()
    except:
        argparser.error("Invalid arguments")
        sys.exit(0)


FanFictionURLCrawler().filter_story_urls(lan=args.lang,
                                      status=args.status,
                                      rating=args.rating,
                                      rate_limit=args.rate_limit,
                                      path_urls= args.output,
                                      base_url = args.base_url,
                                      page=args.page)