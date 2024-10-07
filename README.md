## Semantic Analysis and Role Labeling in News Articles
## Team NLP WIZARDS
## Team Members
 DINESH RAM DANARAJ, 
NISCHITH BAIRANNANAVARA OMPRAKASH,
VISHAL PRAKASH,
MANIKUMAR HONNENAHALLI LAKSHMINARAYANA SWAMY,
NANDAN DAVE

## Project Description
This project focuses on enhancing the semantic understanding of text within news articles through the integration of Semantic Role Labeling (SRL) with advanced Natural Language Processing (NLP) methodologies. The core objective is to extract and categorize actor-action pairs to achieve a human-like interpretation of narratives.

The project refines and utilizes the Knowledge Extraction from Language Model (KELM) dataset and the Bloomberg Quint news dataset, pre-processed using spaCy's capabilities in Named Entity Recognition, Dependency Parsing, and Neural Coreference Resolution. This preparation is crucial for fine-tuning the BERT model to perform multi-label token classification, identifying and classifying actors and their actions within sentences, which significantly improves contextual understanding and the accuracy of semantic role labeling.

Furthermore, an innovative Impact Score method is introduced, incorporating AFINN ratings, textblob, vader, and transformer pipelines to quantify actor-action relevance. The impact of this method is demonstrated through real-world applications, including the analysis of a news article detailing a burglary at Keanu Reeves' residence, showing how different actors and their actions influence the narrative's portrayal.

The results from this project demonstrate the utility of refined annotations in creating more coherent and qualitatively enhanced narratives, essential for accurate and efficient semantic role labeling.

## How to install and run:
## To run this program you need the below package and mentioned version

spacy  - 3.5.0\
charset-normalizer\
chardet\
scipy 1.12.0\
gensim  4.3.2\
nltk  3.8.1\
pandas \
textblob\
vadersentiment\
afinn \
tabulate\
seaborn \
pyparsing \
transformers 4.30.2

Note: Insufficient available in the setup could cause program to stop abruptly.

## Install below spacy models using the command
python3 -m pip install coreferee
python3 -m coreferee install en
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg

## Downloading trained model
Download the trained model available at [model link](https://uflorida-my.sharepoint.com/:f:/g/personal/nischith_bairann_ufl_edu/EsqGDS7mVVxCmLH0T92WGSYByQEgYjqxdV8HQ4xHD_fTlg?e=1Ug0VR)

Place the saved_model and saved_tokenizer folder in project director

## To run program to get top actor rating 
python rate_actor.py --path article.txt -- replace article.txt with file path containing article text
Other setting can be found in config.json file in  "testing_config"

## To run program for generating annotation run
python run_annotation_generator.py
Setting can be found in config.json file under 'annotation'


## To train model run
python model_trainner.py 
Setting can be found in config.json file under 'training_config'


