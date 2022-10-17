import streamlit as st
import pandas as pd
import numpy as np 
import joblib
import pickle
import re
import unidecode
import nltk
import keras

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.title('Categorização de frases multilabel')
st.markdown('Mostrando as categorias classificadas')
Text = st.text_input('Digite a frase e pressione Enter')


@st.cache(allow_output_mutation=True)
def load(tfidf_path, model_path, df_path):
    vectorizer = None
    print('Baixando arquivos necessários...')
    #Carregando Stopwords
    nltk.download('stopwords') 
    stopwords = nltk.corpus.stopwords.words('portuguese')

    print('Carregando Modelos e Datasets...SS')
    #Carregando TF-IDF
    with open(tfidf_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    #Carregando Modelo
    model = keras.models.load_model(model_path)

    #Carregando Pedaço do Dataframe
    df_sample = pd.read_pickle(df_path).columns

    return stopwords, vectorizer, model, df_sample

def pre_process(text, stopwords):
	'''Preprocess input text'''

	text= unidecode.unidecode(re.sub(r'[.,"\'-?:!;]', ' ', ' '.join([word for word in text.split() if word not in (stopwords)])).lower())
	return text

def process_df(vectorizer,text, columns, model):
    singular = vectorizer.transform(pd.Series([text])).toarray()
    resultado = model.predict(singular, verbose=0)
    print('Probabilidades: ', [np.round(x,3) for x in resultado][0])
    result_list = []
    for i in range(len(columns)):
        if (resultado[0][i] > 0.5):
            result_list.append(1)
        else:
            result_list.append(0)

    series = pd.Series(result_list, index = columns)
    df_out = series.to_frame()
    df_out.index.name = 'Categorias'
    df_out.rename(columns={df_out.columns[0]: 'Sim/Não'},inplace=True)
    return df_out

def run():
    stopwords, vectorizer, model, columns = load('tfidf.pickle', 'sequential_model', 'df_sample.pickle')
    print('Processando o texto...')
    text = pre_process(Text, stopwords)
    out = process_df(vectorizer, text, columns, model).reset_index()
    st.write(out)



if __name__ == '__main__':
    run()