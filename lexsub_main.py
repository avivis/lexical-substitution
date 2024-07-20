import sys
from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import numpy as np
import string
import gensim
import transformers 

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    poss = []

    for s in wn.synsets(lemma, pos = pos):
        for l in s.lemmas():
            poss.append(l.name())

    poss = list(set(poss))
    for i in range(0,len(poss)):
        if '_' in poss[i]:
           poss[i] =  poss[i].replace('_',' ')

    return set(filter(lambda a: a != lemma, poss))

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    words = {}
    maxv = 0
    maxk = ''

    for s in wn.synsets(context.lemma, pos = context.pos):
        for l in s.lemmas():
            if l.name() == context.lemma:
                continue
            if l.name() in words:
                words[l.name()]+=l.count()
            else:
                words[l.name()]=l.count()
    for k in words.keys():
        if maxv == 0 or words[k] > maxv:
            maxv = words[k]
            maxk = k

    if '_' in maxk:
        maxk = maxk.replace('_',' ')
        
    return maxk

def wn_simple_lesk_predictor(context : Context) -> str:
    maxv = 0
    maxk = ''
    maxc=-1
    maxl=''

    for syn in wn.synsets(context.lemma, pos = context.pos):
        hypers = ''
        ov = 0
        for item in syn.hypernyms():
            hypers = hypers + ' ' + item.definition() + ' ' + ' ' + (' ').join(item.examples()) + ' '
        res = list(set((syn.definition().strip() + ' ' + (' ').join(syn.examples()).strip() + ' ' + hypers).split(' ')))
        for i in range(0,len(res)):
            if not res[i].isalpha() or res[i] in stopwords.words('english'):
                res[i] = '-1'
        contexts = list(set(((' ').join(context.left_context) + ' ' + (' ').join(context.right_context)).split(' ')))
        for i in range(0,len(contexts)):
            if not contexts[i].isalpha() or contexts[i] in stopwords.words('english'):
                contexts[i] = '-1'
        for res in list(filter(lambda a: a != '-1', res)):
            if res in list(filter(lambda a: a != '-1', contexts)):
                ov += 1
        if ov > maxv:
            for l in syn.lemmas():
                if l.name() != context.lemma:
                    maxv = ov
                    maxk = syn
    if maxv==0:
        for syn in wn.synsets(context.lemma, pos=context.pos):
            for l in syn.lemmas():
                if l.count()>maxc and l.name()!=context.lemma:
                    maxc=l.count()
                    maxl=l.name()
        return maxl    
    for l in maxk.lemmas():
        if l.count()>maxc and l.name() != context.lemma:
            maxc = l.count()
            maxl = l.name()
    return maxl


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    #PART 4
    def predict_nearest(self,context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        words = {}
        maxv = -1
        maxk = ''
        for s in candidates:
            try:
                score = self.model.similarity(context.lemma, s)
            except Exception as e:
                score = 0
            words[s] = score
            if maxv == -1 or score > maxv:
                maxv = score
                maxk = s
        return maxk
    
    def p_6(self, context: Context, top_k=5, similarity_threshold=0.5) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        words = {}
        for s in candidates:
            try:
                score = self.model.similarity(context.lemma, s)
            except Exception as e:
                score = 0
            words[s] = score

        ranked = sorted(words.items(), key=lambda x: x[1], reverse=True)
        selected = [candidate for candidate, score in ranked if score >= similarity_threshold][:top_k]
        best = max(selected, key=lambda x: words[x], default="")
        return best


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    #PART 5
    def predict(self, context: Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        maxs = -1
        best = ""

        tinput = self.tokenizer.encode_plus(context.left_context + [self.tokenizer.mask_token] + context.right_context, return_tensors="tf",
            truncation=True,max_length=512,padding="max_length")
        mask = np.where(tinput['input_ids'][0] == self.tokenizer.mask_token_id)[0]
        if not mask:
            raise ValueError("no mask token")
        pred = self.model.predict(tinput['input_ids'], verbose=0)
        
        for c in candidates:
            try:
                s = pred[0][0][mask][0][self.tokenizer.convert_tokens_to_ids(c)]
            except:
                s = 0
            if s > maxs:
                maxs = s
                best = c
        return best

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    #bert_predictor = BertPredictor()
    #print(get_candidates('slow', 'a'))
    for context in read_lexsub_xml(sys.argv[1]):
        #print(wn_frequency_predictor(context))  # useful for debugging
        # prediction = smurf_predictor(context) 
        #prediction = wn_frequency_predictor(context) 
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest(context) 
        #prediction = bert_predictor.predict(context)
        prediction = predictor.p_6(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
