# IR_Complex_Answer_Retrieval
This project is build by students of the University of Mannheim for the master course Information Retrieval (IE 681).

## Prerequisites

What do you need to run this project?

### Dependencies
- numpy
- nltk (stopwords, PorterStemmer, WordnetLemmatizer)
- pandas
- sklearn
- trectools
- trec-car-tool
- xgboost
### Ranklib:
For L2R, we used the java library by Lemur.
See: https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/

## Explanation
### Main Class
The main class is the workflow.py which computes all scores and run the metrics functions.
The class contains several variables that act as parameters that you can tune.
Additionally, if you want to run a different L2R model, you should change it in the workflow.py in the execute_L2R_task
function.

### Pre-Processing
All pre-processing steps are implemented in the preprocessing.py. It contains a
Preprocess class that do the lemmatization/stemming, stopword removal, ...

### HTMLCreator
This class creates our synthetic HTML Wiki page.

### Retrieval Models
The VSM model TF-IDF is implemented as a class in the tf-idf.py.
The probabilistic BM25 model is implemented as a class in the bm25.py.

### FeatureGenerator
The feature_generator.py contains the FeatureGenerator class. This class provides
functions to generate all the scores for our retrieval models.
This includes the two sub-tasks: 
 1. Scores for query-paragraph retrieval (used as features for L2R)
 2. Scores for paragraph-paragraph retrieval

### GloVe
To cache the glove vectors yourself, you will need the glove.840B.300d.txt or an other
set. In that case change the target file in cache_word_embeddings.py.
The wordembedding.py contains the implementation to parse the pretrained GloVe file and extract
the semantic word embedding vector.
See: https://nlp.stanford.edu/projects/glove/

### Create Input for L2R
The create_input_for_L2R.py contains functions that takes out retrieval models' scores
and convert them into a file format readable as features for L2R.

### Scorer
Our scorer implementation can be found in the metrics_calculate.py. We implemented for our data structures scorers for Accuracy, Precision, Recall, F1-Score in the
Standard class. The metrics class contains scorers for Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR).
