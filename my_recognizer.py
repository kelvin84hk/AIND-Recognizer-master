import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    for i in range(test_set.num_items):

      testX,textLen=test_set.get_item_Xlengths(i)
      tempProb={}
      tempPre=''
      maxLog=float('-inf')
      for mkey, M in models.items():
        try:
          prob=M.score(testX,textLen)
          tempProb[mkey]=prob
          if prob>maxLog:
            maxLog=prob
            tempPre=mkey
        except:
          continue    
        
      probabilities.append(tempProb)
      guesses.append(tempPre)
      
    return probabilities,guesses
