import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

class Word_Processing1:

    english_stopwords = set(stopwords.words("english"))
    WORD_REGEX = '[^A-Za-z0-9]+'
    HTTP_REGEX = r"http\S+"
    DIG_REGEX = r"\d+"
    ATTHERATE_REGEX = r"@\S+"
    HASHTAG_REGEX = r"#\S+"
    
    #@staticmethod
    #def clean_line(lines):
    #    words = map(lambda x : re.sub('\n','',x),lines)
    #    words = list(map(lambda x:  re.sub(Word_Processing1.WORD_REGEX, '', x), words))
    #    return words

    @staticmethod
    def clean_word(word):
        word = word.strip()
        word = word.lower()
        word = re.sub(Word_Processing1.WORD_REGEX, '', word)
        word = re.sub(Word_Processing1.HTTP_REGEX, '', word)
        word = re.sub(Word_Processing1.DIG_REGEX, '', word)
        word = re.sub(Word_Processing1.ATTHERATE_REGEX, '', word)
        word = re.sub(Word_Processing1.HASHTAG_REGEX, '', word)
        if word not in Word_Processing1.english_stopwords:
            return word
        else:
            return ''