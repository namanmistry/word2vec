
from flask import Flask, render_template,request
import pickle
from tensorflow.keras.models import load_model
from scipy import spatial


app = Flask(__name__,template_folder='web_pages')

model = load_model('https://drive.google.com/file/d/1SJjb0cncNwrEiTxIDasTVTGZ_fhZh0L5/view?usp=sharing')
model.pop()
with open('text_to_seq.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('inference_dict_for_word2vec.pkl', 'rb') as f:
    inference_dict = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['post'])
def get_answer():
    text = request.form['input_data']
    similarities = []
    ans = []
    seq = tokenizer.texts_to_sequences([text])
    pred = model.predict(seq)
    for j,i in enumerate(model.layers[0].get_weights()[0]):
        similarities.append([(1 - spatial.distance.cosine(pred[0], i)),j])
    similarities = sorted(similarities,key=lambda x: x[0],reverse=True)
    for i in range(10):
        ans.append((inference_dict[similarities[i][1]],similarities[i][0]))
    return f"{ans}"

if __name__ == '__main__':
    app.run(debug=True)