from ai.nltk_utils import tokenize, stem

def get_number_from_text(text):
    units = [
        'нол', 'один', 'два', 'три', 'четыр', 'пят', 'шест', 'сем', 'восем', 'девя', 'деся', 
        'одиннадца', 'двенадца', 'тринадца', 'четырнадца', 'пятнадца', 'шестнадца', 'семнадца', 'восемнадца', 'девятнадца'
    ]

    tens = ['двадца', 'тридца', 'сорок', 'пятьдес', 'шестьдес', 'семьдес', 'восемьдес', 'девян']

    hundreds = ["сто", "двест", "трист"]


    result = 0
    
    for word in tokenize(text):
        stemmed_word = stem(word)
        
        if stemmed_word in units:
            result += units.index(stemmed_word)
        elif stemmed_word in tens:
            result += (tens.index(stemmed_word) + 2) * 10
        elif stemmed_word in hundreds:
            result += (hundreds.index(stemmed_word) + 1) * 100
            
            
    return result

