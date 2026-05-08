"""
En este archivo estan los tokenizers y stemmers
"""
from nltk.stem.snowball import SnowballStemmer
from typing import List, Callable, Optional, Pattern
import re

stemmer = SnowballStemmer("spanish")


def stem(tokens: List[str]) -> List[str]:
    """
    Transforma mediante un stemmer a una secuencia de tokens.
    :param tokens: Una secuencia de tokens.
    :return La secuencia de tokens transformada por el stemmer.
    """
    global stemmer
    return [stemmer.stem(w.lower()) for w in tokens]


def tokenizador(token_regex: Optional[Pattern] = None) -> Callable[[str],List[str]]:
    """
    :param token_regex: Una expresion regular que define que es un token
    :return: Una funcion que recibe un texto y retorna el texto tokenizado.
    """
    if token_regex is None:
        # definicion de que es un token: una letra seguida de letras y numeros
        token_regex = r"[a-zA-ZâáàãõáêéíóôõúüÁÉÍÓÚñÑçÇ][0-9a-zA-ZâáàãõáêéíóôõúüÁÉÍÓÚñÑçÇ]+"
    token_pattern = re.compile(token_regex)
    return lambda doc: token_pattern.findall(doc)


def tokenizador_con_stemming(token_regex: Optional[Pattern] = None) ->  Callable[[str], List[str]]:
    """
    :param token_regex: Una expresion regular que define que es un token
    :return: Una funcion que recibe un texto y retorna el texto tokenizado y transformado por un stemmer en español.
    """
    tokenizer = tokenizador(token_regex)
    return lambda doc: stem(tokenizer(doc))
