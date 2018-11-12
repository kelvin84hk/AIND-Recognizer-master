import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model=[]
        minM=float('inf')
        minI=0
        c=-1
        for i in range(self.min_n_components,self.max_n_components+1):
            try:
                
                hmm=self.base_model(i)
                LogL=hmm.score(self.X,self.lengths)
                f=hmm.n_features
                p = i ** 2 + 2 * f * i - 1
                BI=-2*LogL+np.log(len(self.X))*p
                model.append((hmm,BI))
                c=c+1
                if model[c][1]<minM:
                    minM=model[c][1]
                    minI=c
                    
            except:
                
                continue

        if len(model)>0:        
            return model[minI][0]
        else:
            return None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        allwords=self.words.keys()
        model=[]
        maxM=float('-inf')
        maxI=0 
        c=-1
        for i in range(self.min_n_components,self.max_n_components+1):
            try:
                hmm=self.base_model(i)
                Log=hmm.score(self.X,self.lengths)
                otherLog=[]
                for otherw in allwords:
                    if otherw != self.this_word:
                        X2, len2=self.hwords[otherw]
                        otherLog.append(hmm.score(X2, len2))
                                                          
                DIC=Log-np.average(otherLog)
                model.append((hmm,DIC))
                c=c+1
                if  model[c][1]>maxM:
                    maxM=model[c][1]
                    maxI=c
            except:
                continue

        
        if len(model)>0:        
            return model[maxI][0]
        else:
            return None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        myS=self.sequences
        split_method = KFold(n_splits=2)
        model=[]
        maxM=float('-inf')
        maxI=0
        c=-1
        for i in range(self.min_n_components,self.max_n_components+1,1):
    
            if len(myS)>1:
                count=0
                score=0
                for cv_train_idx, cv_test_idx in split_method.split(myS):
                    trainX, trainLen=combine_sequences(cv_train_idx,myS)
                    testX, testLen=combine_sequences(cv_test_idx,myS)   
                    try:
                        hmm=self.base_model(i)
                        score+=hmm.score(testX,testLen)
                        count+=1
                    except:
                        
                        continue
            
                
                if count!=0:        
                    model.append((hmm,score/count))
                    c=c+1
            else:
                try:
                    hmm=self.base_model(i)
                    score=hmm.score(self.X,self.lengths)
                    model.append((hmm,score))
                    c=c+1
                except:
                    continue
            
            if model[c][1]>maxM:
                maxM=model[c][1]
                maxI=c
              
        if len(model)>0:        
            return model[maxI][0]
        else:
            return None
