import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')


def clean_text(text):
    text = re.sub("\n", ' ', str(text))
    text = re.sub("\s+", ' ', text)
    text = text.lower()
    text = re.sub("\w+@\w+\.\w+?", '', text)
    text = re.sub("\'", ' ', text)
    text = re.sub("[\-\+\,\.\(\)\;\:\<\>\?\^\@\#\*\[\]\{\}\"\$\%\_\&]", '', text)
    text = re.sub("[2345678]", '', text)
    text = re.sub("\t", ' ', text)
    text_clean = re.sub("\s+", ' ', text)
    return text_clean


def remove_stopwords(serie):
    stop_words = stopwords.words('english')
    stop_words.extend(['r', 'v', 'x'])
    return [word for word in serie if word not in stop_words]


def lemmatization(serie):
    return [WordNetLemmatizer().lemmatize(str(word)) for word in serie]
