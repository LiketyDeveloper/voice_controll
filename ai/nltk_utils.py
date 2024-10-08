import nltk
# nltk.download('punkt_tab')
from nltk.stem.snowball import SnowballStemmer
import numpy as np

 
def tokenize(text: str) -> list:
    """
    Split a string into meaningful units (tokens). These units are separated by whitespace, punctuation, or other special characters.

    ### For example:
    "hello world, how are you?" 
    
    -> 
    
    ["hello", "world,", "how", "are", "you", "?"]

    **return**: list of tokens
    """
    return nltk.word_tokenize(text)


def stem(word: str) -> str:
    """
    Returns the root form of a word, also known as the word's lemma.

    ### For example:
    "running" -> "run"

    **return**: root word
    """
    stemmer = SnowballStemmer("russian")
    return stemmer.stem(word)
    


def bag_of_words(tokenized_words: list, all_words: list) -> np.ndarray:
    """
    Creates the bag of words based on given list of tokenized words and list of all words.
    
    ### For example: 
    ["hello", "world", "how", "are", "you"]
    
    "How are you" -> [0, 0, 1, 1, 1]
    
    **return**: bag of words
    """
    
    tokenized_words = [stem(w) for w in tokenized_words]
    
    # bag = np.zeros(len(all_words), dtype=np.float32)
    bag = [0.0 for _ in range(len(all_words))]
    for idx, w in enumerate(all_words):
        if w in tokenized_words:
            bag[idx] = 1.0
            
    return bag        