Avighna Suresh (as6469)

# Necessary packages:
NLTK: `$ pip install nltk`

gensim: `pip install gensim`

BERT: `pip install transformers`

# Files:
`lexsub_trial.xml` 
- input trial data containing 300 sentences with a single target word each.
  
`gold.trial` 
- gold annotations for the trial data (substitues for each word suggested by 5 judges).
  
`lexsub_xml.py` 
- an XML parser that reads lexsub_trial.xml into Python objects.
  
`lexsub_main.py` - 
- Part 1: Candidate Synonyms from WordNet - takes a lemma and part of speech ('a','n','v','r' for adjective, noun, verb, or adverb) as parameters and returns a set of possible substitutes.
- Part 2: WordNet Frequency Baseline - takes a context object as input and predicts the possible synonym with the highest total occurence frequency (according to WordNet)
- Part 3: Simple Lesk Algorithm - uses Word Sense Disambiguation (WSD) to select a synset for the target word, then returns the most frequent synonym from that synset as a substitute
- Part 4: Most Similar Synonym - returns the most similar synonym using Word2Vec
- Part 5: Using BERT's masked language model - typically, BERT is used to compute contextualized word embeddings, or a dense sentence representation. In these applications, the masked LM layer, which uses the contextualized embeddings to predict masked words, is removed and only the embeddings are used. 

`score.pl` 
- The scoring script provided for the SemEval 2007 lexical substitution task.
