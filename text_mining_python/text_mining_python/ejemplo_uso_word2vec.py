# -*- coding: utf-8 -*-

# Ejemplo de como transformar un parrafo a un vector usando los vectores de palabras de word2vec.
# Puede conseguir un modelo en español de word2vec en https://github.com/aitoralmeida/spanish_word2vec
# u otro mejor, mas grande en https://github.com/dccuchile/spanish-word-embeddings ; descarguelo en formato "vec". Este modelo utiliza unos 3 GB de RAM.
# Solo necesita 1 de los 2 modelos.
# MUY IMPORTANTE: En ambos casos vea las reglas de tokenizacion del modelo de word2vec correspondiente. Conservaron mayusculas? acentos?

from gensim.models import KeyedVectors
from word2vec import MeanEmbeddingVectorizer
from tokenizers import tokenizador, tokenizador_con_stemming
from gensim.test.utils import datapath

  
# ALTERNATIVA 1: Leer los vectores de palabras de "aitoralmeida" . Las palabras se asumen en minusculas, con acentos
word_vectors = KeyedVectors.load('complete.kv', mmap='r')
# ALTERNATIVA 2: Leer los vectores de palabras de la univ. de chile en formato "vec" (texto) : palabras en minusculas, secuencias de digitos fueron reemplazados por 1 "0" (cero), palabras de mas de 3 letras terminadas en digitos fueron eliminadas 
word_vectors = KeyedVectors.load_word2vec_format(datapath('word2vec_es.vec'), binary=False) 
# ... y guardarlo para despues poder cargarlo con load() como el 1er modelo "aitoralmeida"
word_vectors.save('word2vec_es.kv')

# cualquiera de las alternativas usadas, vectorizar una frase usando el centroide de los embeddings de los tokens de la frase.
vectorizer = MeanEmbeddingVectorizer(word_vectors, tokenizer=tokenizador())
# matriz contendra 1 vector por  cada frase
matriz = vectorizer.transform(["El que depositó dólares recibirá dólares","Estamos condenados al éxito"])
print(matriz)
