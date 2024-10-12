<<<<<<< HEAD
import pymorphy3
morph = pymorphy3.MorphAnalyzer()

print(["привет", "пока"] == ["привет", "пока"])
=======
import os
import json
from tqdm import tqdm

from util import get_path
from util.nltk import tokenize, stem, get_stopwords, is_number
from sklearn.feature_extraction.text import CountVectorizer

import pymorphy3

ANNOTATIONS_PATH = get_path("ai", "dataset", "annotation")
annotations = []


def preprocess_text(text: str) -> list:
    """Returns a list of stemmed tokens from given string"""
    result = [stem(token) 
              for token in tokenize(text) 
              if morph.parse(token)[0].tag.POS != 'NUMR'
              and token not in get_stopwords()
              ]
    
    return " ".join(result)

for filename in os.listdir(ANNOTATIONS_PATH):
    if filename.endswith(".json"):
        with open(os.path.join(ANNOTATIONS_PATH, filename), "r", encoding="utf-8") as file:
            data = json.load(file)
            annotations.extend(data)

morph = pymorphy3.MorphAnalyzer()

docs = []
for annotation in tqdm(annotations):
    text = preprocess_text(annotation["text"])
    docs.append(text)
    
docs = sorted(set(docs))
    
cv = CountVectorizer(ngram_range=(2,2))
X = cv.fit_transform(docs)
X = X.toarray()[0]



text = "межвагонное пространство"
print(text)
text = preprocess_text(text)
print(text)
text_vector = cv.transform([text]).toarray()[0]
print(text_vector)
print(sorted(cv.vocabulary_.keys()))
>>>>>>> 56d82a7 (Refactored AI training code, added metrics, moved the task processing to other thread)
