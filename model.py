from flask import Flask, request, jsonify
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

model_path = r'C:/versionTest/악플감지모델'

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU 사용
    return_all_scores=True,
    function_to_apply='sigmoid'
)

@app.route('/', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        text = data['text']

        # 입력 문장에 대한 결과 예측
        results = pipe(text)

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
