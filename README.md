# Harry Potter and the Action Prediction Challenge from Natural Language

This repository contains the code to download the HPAC corpus and a set of simple baselines. 

## Preriquisites
- Python 2.7
- requests 2.21.0
- bs4 4.7.1
- ntlk 3.4
- hashedindex 0.4.4
- numpy 1.16.2
- tensorflow-gpu 1.13.1
- keras 2.2.4
- sklearn 0.20.3
- prettytable 0.7.2
- matplotlib 2.2.4
- tqdm
- stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
- We include together with our code a version of the crawler https://github.com/smilli/fanfiction
- python-tk

## Installation 

We recommend to create a virtualenv (`virtualenv hpac`) so these packages do not interfere with previous versions that you might have installed in your machine.

After activating the virtualenv, execute the file `install.sh` to automatically install the mentioned dependencies (tested on Ubuntu 18.04 64 bits).

## Building HPAC

The file `resources/hpac_urls.txt` contains the URLs of the fanfiction stories that we used to build HPAC.

> **NOTE:** Unfortunately, some stories might be deleted by users or admins after they have been published and completed, so not being able to rebuild the 100% of the corpus is a possibility.

> **NOTE:** Also, some stories might have been modified after the corpus was created. This will result in the scripts in charge to generate HPAC not being able to retrieve some samples.

>**NOTE:** This corpus is built in an automatic way and we have not censored the content of the stories. Some of them might contain innapropiate content (e.g. sexual related content). 


### Crawling the data
First, to crawl the fanfiction use the script `scraper.py`:

```
python scraper.py --output resources/fanfiction_texts/ --rate_limit 2 --log scraper.log --url resources/hpac_urls.txt
```

- `--output` The directory where each fanfiction story will be written down (the name of each file will be the ID of the story).
- `--rate_limit` How fast to crawl fanfiction (in number of seconds). To respect ToS, this limit should correspond to the approximate speed you could manually crawl the stories. The value used in the example is illustrative.
- `--url` The text file containing the URLs to crawl (e.g. `resources/hpac_urls.txt`).
- `--log` The path file to log the URLs that could not be retrieved due to some issue.

According to https://github.com/smilli/fanfiction, the rate limit is set in order to comply with the fanfiction.net terms of service:

> E. You agree not to use or launch any automated system, including without limitation, "robots," "spiders," or "offline readers," that accesses the Website in a manner that sends more request messages to the FanFiction.Net servers in a given period of time than a human can reasonably produce in the same period by using a conventional on-line web browser.

### Tokenizing and creating and inverted index

Second, build an index (and a tokenizer) using the script `index.py`. This is done to then be able to quickly create different versions of the corpus using different snippet lengths.


```
python index.py --dir resources/fanfiction_texts/ --spells resources/hpac_spells.txt --tools resources/ --stanford_jar resources/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar --dir_tok resources/fanfiction_texts_tok/
```

- `--dir` The directory containing the fanfiction stories crawled by the script `scraper.py`.
- `--dir_tok` The output directory where to store the tokenized stories.
- `--spells` The file containing the spells to take into account (`hp_spells.txt`).
- `--tools` The output directory where to store the `index` and the `tokenizer` needed to create HPAC.
- `--stanford_jar` The path to `stanford-corenlp-3.8.0.jar`


### Create HPAC
Finally, we can create a version of HPAC using a snnipet of size *x* (e.g. 128) with the script `create_hpac.py`:

```
python create_hpac.py --dir_stories_tok resources/fanfiction_texts_tok/ --output hpac_corpus/ --window_size 128 --index resources/ff.index --hpac_train resources/hpac_training_labels.tsv --hpac_dev  resources/hpac_dev_labels.tsv --hpac_test  resources/hpac_test_labels.tsv
```

- `--dir_stories_tok` The ath to the directory containing the tokenized fanfiction
- `--output` The path to the directory where to store HPAC
- `--window_size` An integer with the size of the snippet (number of tokens)
- `--index` The path to `ff.index` (created in the previous step with `index.py`) 
- `--hpac_train` The file that will contain the IDS of the training samples `resources/hpac_training_labels.tsv`
- `--hpac_dev` The file that will contain the IDS of the dev samples `resources/hpac_dev_labels.tsv`
- `--hpac_test` The file that will contain the IDS of the test samples `resources/hpac_test_labels.tsv`

The scripts generate three files: `hpac_training_X.tsv`, `hpac_dev_X.tsv`, and `hpac_test_X.tsv`, where X is the size of the snippet. This is the HPAC corpus.

### Checking missing elements

As said before, some stories might be deleted from the fanfiction web site or updated, turning into invalid IDS for that particular story. To check how many elements are missing, you can use the script `checker.py`:

```
python checker.py --input hpac_corpus/hpac_dev_128.tsv --labels resources/hpac_dev_labels.tsv
```

- `--input` The path to the generated version of a training, dev or test set
- `--labels` The file containing the IDS of the training/dev/test samples (e.g. `resources/hpac_dev_labels.tsv`)

## Getting additional links to crawl recently created FanFiction

If you want to create a larger set, or simply use Harry Potter fanfiction (or other fanfiction) for other purposes, you can collect your own fan fiction URL links (users create new stories daily) and then run the previous scripts accordingly.

```
python get_fanfiction_links.py --base_url https://www.fanfiction.net/book/Harry-Potter/ --lang en --status complete --rating all --page 1 --output new_fanfiction_urls.txt --rate_limit 2
```

- `--base_url` The URL from where to download fanfiction (we used *https://www.fanfiction.net/book/Harry-Potter/* )
- `--lang` Download stories written in a given language (we used *en*)
- `--status` Download fanfiction with a certain status (we used *completed*)
- `--rating` Download fanfiction with a certain rating (we used *all*)
- `--rate_limit` Makes a request every *x* seconds
- `--page` Download links from page *x*
- `--output` The path where to write the URLS



## Train a model

You can train your model(s) using the `run.py` script:

```
python run.py --training hpac_corpus/hpac_training_128.tsv --test hpac_corpus/hpac_dev_128.tsv --conf resources/configuration.conf --model LSTM --S 2 --gpu 1 --timesteps 128 --dir models/
```

- `--training` The path to the training file
- `--test` The path to the dev set during training
- `--dir` The path to the directory where to store/load the models
- `--conf` The path to the configuration file that contains the hyperparameters for the different models (e.g.  `resources/configuration.conf`)
- `--model` The architecture of the model `[MLR, MLP, CNN, LSTM]`
- `--gpu` The id of the GPU to be used
- `--timesteps` This value should match the size of the snnipet window of the version of HPAC you are using
- `--S` The number of models to train (we used *5* in our experiments). 

Each trained model will be named by `HP_[MLR,MLP,CNN,LSTM]_timesteps_X`, where X is the value of *n* trained model (e.g. `HP_LSTM_128_2`). 


## Run a model

You can run your trained model(s) using `run.py` as well

```
python run.py  --test hpac_corpus/hpac_test_128.tsv --conf resources/configuration.conf --model LSTM --S 5 --predict --model_params models/HP_LSTM_128.params --model_weights models/HP_LSTM_128.hdf5 --gpu 1 --timesteps 128 --dir models/
```

- `--predict` Flag to indicate the script we are on testing.
- `--test` The path to the test set.
- `--conf` The path to the configuration file.
- `--S` To evaluate the first *n* models created during training.
- `--model` The architecture of the model `[MLR, MLP, CNN, LSTM]`.
- `--model_params` The path to the parameters file to be used by the model (ignoring the index that indicates that was the *n* trained model in the previous step).
- `--model_weights` The path to the weights file to be used by the model  (ignoring the index that indicates that was the *n* trained model in the previous step).
- `--timestep` Number of timesteps (needed for sequential models).

# Citation

David Vilares and Carlos Gómez-Rodríguez. Harry Potter and the Action Prediction Challenge from Natural Language. 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics. To appear.

# Contact
If you have any suggestion, inquiry or bug to report, please contact david.vilares@udc.es



