# NLP Land
A repository dedicated to Natural Language Processing.

It includes projects and code examples related to NLP tasks and methods.

## Contents
- [NLP Land](#nlp-land)
  - [Contents](#contents)
  - [Text Summarization](#text-summarization)
    - [With Transformers](#with-transformers)
    - [With Simplet5](#with-simplet5)
    - [Using spaCy](#using-spacy)
  - [Topic Modeling/classification](#topic-modelingclassification)

---


## Text Summarization
### [With Transformers](https://github.com/Dipankar-Medhi/nlpLand/tree/main/text_summarization/text-summarization-transformers)



```
Input Text: 
"""Around 65% (2.37 lakh) of all road accidents in India took place on straight roads in 2020, government data has shown. 
Such accidents killed over 85,000 people, 
the 'Road accidents in India- 2020' report released by the government this year, said. 
Over 72% of accidents and 67% of fatalities took place under sunny or clear weather in 2020, it added."""

Summarized text: 
65% (2.37 lakh) of all road accidents in India  took place on straight roads in 2020 . Such accidents killed over 85,000 people, government data shows .

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
At least six people died and 71 others were hospitalised in the last three days due to diarrhoea after allegedly drinking contaminated water from open sources in several villages in Odisha's Rayagada district, officials said on Saturday. A team of 11 doctors visited the affected villages and collected water and blood samples and sent those for examination.

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
Whilst the crypto markets have taken a pretty public downturn of late, this is a great time to experiment and build your own web3 projects. Ether, and almost all other cryptocurrencies, are cheaper to get your hands on, gas fees (the transaction fees paid when using blockchain networks) are much lower than their 2021 highs and now is your chance to learn a new skill that could be very valuable going forward.

Output summary:
Ether, and almost all other cryptocurrencies, are cheaper to get your hands on, gas fees (the transaction fees paid when using blockchain networks) are much lower than their 2021 highs and now is your chance to learn a new skill that could be very valuable going forward., Whilst the crypto markets have taken a pretty public downturn of late, this is a great time to experiment and build your own web3 projects.
```

## Topic Modeling/classification
- [Supervised classification](https://github.com/Dipankar-Medhi/nlpLand/blob/main/topic_classification/topic_classification_supervised_ML.ipynb) &
- [Unsupervised classification (Clustering)](https://github.com/Dipankar-Medhi/nlpLand/blob/main/topic_classification/topic_classification(clustering)_unsupervised_ML.ipynb)

