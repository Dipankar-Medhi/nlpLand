# NLP Land
A repository dedicated to Natural Language Processing.

It includes projects and code examples related to NLP tasks and methods.

## Contents
- [NLP Land](#nlp-land)
  - [Contents](#contents)
  - [Some nlp techniques](#some-nlp-techniques)
    - [Extract Noun](#extract-noun)
    - [Similarity between words](#similarity-between-words)
    - [Part of Speech tagging](#part-of-speech-tagging)
    - [Entities extraction](#entities-extraction)
  - [Text Summarization](#text-summarization)
    - [With Transformers](#with-transformers)
    - [With Simplet5](#with-simplet5)
    - [Using spaCy](#using-spacy)
  - [Topic Modeling/classification](#topic-modelingclassification)

---

## Some nlp techniques
### Extract Noun
```python
blob = TextBlob("Rahul is a great Machine Learning Engineer. He is specialized in Natural Language Processing.")
blob.noun_phrases

# output ---
# WordList(['rahul', 'machine learning engineer', 'language processing'])
```
### Similarity between words
**Cosine Similarity**
```python
documents = ( 
"I like Apple",
"I am exploring Apple devices",
"I am a beginner in Apple development", 
"I want to work for Apple", 
"I like Apple products"
)
# code ...
# similarity of 1st sentence with the rest
cosine_similarity(tfid_matrix[0:1], tfid_matrix)

# output ---
# array([[1.        , 0.14284054, 0.12305308, 0.11786255, 0.68374784]])
```
### Part of Speech tagging
```python
text = "I love programming. I love building softwares."
# Tagging all the tokens
for token in tokens:
  words = nltk.word_tokenize(token)
  words = [w for w in words if not w in stop_words]
  # POS tagger
  tags = nltk.pos_tag(words)

print(tags)
# output ---
# [('I', 'PRP'),
# ('love', 'VBP'),
# ('building', 'VBG'),
# ('softwares', 'NNS'),
# ('.', '.')]
```
<details>
<summary>Some common tags and their meaning</summary>

- CC coordinating conjunction
- CD cardinal digit
- DT determiner
- EX existential there (like: “there is” ... think of it like 
“there exists”)
- FW foreign word
- IN preposition/subordinating conjunction
- JJ adjective ‘big’
- JJR adjective, comparative ‘bigger’
- JJS adjective, superlative ‘biggest’
- LS list marker 1)
- MD modal could, will
- NN noun, singular ‘desk’
- NNS noun plural ‘desks’107
- NNP proper noun, singular ‘Harrison’
- NNPS proper noun, plural ‘Americans’
- PDT predeterminer ‘all the kids’
- POS possessive ending parent’s
- PRP personal pronoun I, he, she
- PRP$ possessive pronoun my, his, hers

- RB adverb very, silently
- RBR adverb, comparative better
- RBS adverb, superlative best
- RP particle give up
- TO to go ‘to’ the store
- UH interjection
- VB verb, base form take
- VBD verb, past tense took
- VBG verb, gerund/present participle taking
- VBN verb, past participle taken
- VBP verb, sing. present, non-3d take
- VBZ verb, 3rd person sing. present takes
- WDT wh-determiner which
- WP wh-pronoun who, what
- WP$ possessive wh-pronoun whose
- WRB wh-adverb where, when

</details>

### Entities extraction
```python
from nltk import ne_chunk
# NER
ne_chunk(nltk.pos_tag(word_tokenize(text)), binary=False)
# output ---
# Tree('S', [Tree('GPE', [('Rahul', 'NNP')]), ('is', 'VBZ'), ('a', 'DT'), ('very', 'RB'), ('good', 'JJ'), ('footballer', 'NN'), ('.', '.'), ('He', 'PRP'), ('wants', 'VBZ'), ('to', 'TO'), ('play', 'VB'), ('for', 'IN'), ('his', 'PRP$'), ('country', 'NN'), ('.', '.')])
```

---


## Text Summarization
### [With Transformers](https://github.com/Dipankar-Medhi/nlpLand/tree/main/text_summarization/text-summarization-transformers)



```
Input Text: 
Around 65% (2.37 lakh) of all road accidents in India took place on 
straight roads in 2020, government data has shown. 
Such accidents killed over 85,000 people, 
the 'Road accidents in India- 2020' report released by the
government this year, said. 
Over 72% of accidents and 67% of fatalities took place under sunny 
or clear weather in 2020, it added.

Summarized text: 
65% (2.37 lakh) of all road accidents in India  took place on 
straight roads in 2020 . Such accidents killed over 85,000 people, 
government data shows .

```


**Model**:
```python

model_name = "sshleifer/distilbart-cnn-12-6"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

result = summarizer2(text, max_length=130, min_length=30, do_sample=False)

```

### [With Simplet5](https://github.com/Dipankar-Medhi/nlpLand/blob/main/text_summarization/text-summarization-transformers/text-summarization-T5.ipynb)

```
Input text:
At least six people died and 71 others were hospitalised in the 
last three days due to diarrhoea after allegedly drinking 
contaminated water from open sources in several villages in 
Odisha's Rayagada district, officials said on Saturday. A team of 
11 doctors visited the affected villages and collected water and 
blood samples and sent those for examination.

Summarized text:
6 dead, 71 others hospitalised after drinking contaminated water
```

**Model**
```python
model.from_pretrained(model_type='t5', model_name='t5-base')
```

### [Using spaCy](https://github.com/Dipankar-Medhi/nlpLand/tree/main/text_summarization/text-summarization-spacy)

```
Input sentence:
Whilst the crypto markets have taken a pretty public downturn of
 late, this is a great time to experiment and build your own web3 
 projects. Ether, and almost all other cryptocurrencies, are 
 cheaper to get your hands on, gas fees (the transaction fees paid 
 when using blockchain networks) are much lower than their 2021 
 highs and now is your chance to learn a new skill that could be 
 very valuable going forward.

Output summary:
Ether, and almost all other cryptocurrencies, are cheaper to get
 your hands on, gas fees (the transaction fees paid when using 
 blockchain networks) are much lower than their 2021 highs and now 
 is your chance to learn a new skill that could be very valuable 
 going forward., Whilst the crypto markets have taken a pretty 
 public downturn of late, this is a great time to experiment and 
 build your own web3 projects.
```

## Topic Modeling/classification
- [Supervised classification](https://github.com/Dipankar-Medhi/nlpLand/blob/main/topic_classification/topic_classification_supervised_ML.ipynb) &
- [Unsupervised classification (Clustering)](https://github.com/Dipankar-Medhi/nlpLand/blob/main/topic_classification/topic_classification(clustering)_unsupervised_ML.ipynb)

