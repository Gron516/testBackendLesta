from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            text = file.read().decode('utf-8')
            return process_text(text)
    return render_template('index.html')

def process_text(text):
    # Разделяем текст на слова
    words = text.split()
    
    # Создаем DataFrame для подсчета TF
    df = pd.DataFrame(words, columns=['word'])
    tf_counts = df['word'].value_counts().reset_index()
    tf_counts.columns = ['word', 'tf']
    
    # Вычисляем IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    
    idf_values = vectorizer.idf_
    idf_dict = dict(zip(vectorizer.get_feature_names_out(), idf_values))
    
    # Добавляем IDF в DataFrame
    tf_counts['idf'] = tf_counts['word'].map(idf_dict)
    
    # Сортируем по убыванию IDF и берем топ-50 слов
    result_df = tf_counts.sort_values(by='idf', ascending=False).head(50)
    
    return render_template('result.html', tables=[result_df.to_html(classes='data')], titles=result_df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)