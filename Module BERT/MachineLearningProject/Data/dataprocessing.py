# Import Auxiliary modules
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from spellchecker import SpellChecker
import random


class TextCleaner:
    def __init__(self, language='english'):
        # Download the stopwords resource
        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words(language))

    def clean_text(self, text):
        """
        Convert text to lowercase and remove punctuation, numbers, and extra spaces.
        """
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                return ""
        text = text.lower()
        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # remove numbers
        text = re.sub(r'\d+', '', text)
        # remove extra space
        text = ' '.join(text.split())
        return text

    def remove_stopwords(self, text):
        """
        Remove stop words from text
        """
        tokens = text.split()
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(filtered_tokens)

    def preprocess(self, text):
        """
       Comprehensive application of cleaning and stop word removal
        """
        try:
            cleaned = self.clean_text(text)
            result = self.remove_stopwords(cleaned)
            return str(result)
        except Exception:
            return ""

    def tokenize(self, text):
        """
        Split by space
        """
        return str.split(text)

    def correct_text(self, text):
        '''
        Spell Checker
        '''
        spell = SpellChecker()
        tokens = text.split()
        corrected_tokens = [spell.correction(token) for token in tokens]
        return " ".join(corrected_tokens)

    nltk.download('wordnet')

    def synonym_replace(self,text, n=2):
        words = text.split()
        new_words = words.copy()
        random_indices = list(range(len(words)))
        random.shuffle(random_indices)

        num_replaced = 0
        for idx in random_indices:
            synonyms = wordnet.synsets(words[idx])
            if not synonyms:
                continue
            lemmas = synonyms[0].lemmas()
            synonym_words = [lemma.name() for lemma in lemmas if lemma.name().lower() != words[idx].lower()]
            if synonym_words:
                new_words[idx] = random.choice(synonym_words)
                num_replaced += 1
            if num_replaced >= n:
                break
        return " ".join(new_words)



