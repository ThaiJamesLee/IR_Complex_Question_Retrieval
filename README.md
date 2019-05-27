# IR_Complex_Answer_Retrieval

## Main Class
The main class is the workflow.py which computes all scores and run the metrics functions.
The class contains several variables that act as parameters that you can tune.
Additionally, if you want to run a different L2R model, you should change it in the workflow.py in the execute_L2R_task
function.

## HTML Creator
This class creates our synthetic HTML Wiki page.

## GloVe
To cache the glove vectors yourself, you will need the glove.840B.300d.txt or an other
set. In that case change the target file in cache_word_embeddings.py.

## Ranklib:

more reference: https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/