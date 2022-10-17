BLESSED README FILE

Saudações, esse projeto tem como objetivo realizar uma classificação multilabel em uma dataset com 500+ linhas conendo frases não-nulas.
Esse é um problema multilabel, ou seja, cada frase pode ser classificada em mais de uma categoria, sendo elas:
*   Educação
*   Finanças
*   Industrias
*   Orgão Público
*   Varejo

Foram utilizadas nesse projeto, técnicas de NLP, como TFIDF, diversos tipos de modelos sklearn e um modelo de Rede Neural utilizando Keras/Tensorflow
A análise exploratório e todo o desenvolvimento pode ser encontrado no *dev-nlp-phrases.ipynb*

A versão do python recomendada é a 3.9, versões divergentes podem ocasionar em incompatibilidade com o modelo utilizado.

Os requisitos deste repositório podem ser encontrados no arquivo *requirements.txt* e instalados utilizando:

<pre><code>pip3 install -r requirements.txt
</code></pre>

Para executar e testar as frases usando uma interface HTML foi utilizado o StreamLit, para executá-lo, execute na raiz do projeto o código:

<pre><code>streamlit run app.py
</code></pre>

Após abrir a interface, siga as instruções e divirta-se!!

![alt text](anya.gif "heh")
