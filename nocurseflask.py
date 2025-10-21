from flask import Flask, request, jsonify,request, render_template
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
from flask_cors import CORS
import re
from kiwipiepy import Kiwi
kiwi = Kiwi()
import openai

app = Flask(__name__)
CORS(app) 

def chat3_sentence(sentence, tag):
    # API 키를 여기에 입력하세요.
    openai.api_key = "api_key"

    question = f" \"{sentence}\"을 {tag}형으로 바꿔줘(1개).{tag}형으로 바꾼 결과를 대괄호안에 나타내줘"
    print(f"질문: {question}")  # 사용자의 질문을 출력합니다.

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        # 답변을 추출합니다.
        answer = completion.choices[0].message['content']
        # 답변을 반환합니다.
        return answer

    except Exception as e:
        # 오류가 발생한 경우, 오류 메시지를 반환합니다.
        return f"오류: {str(e)}"

remove_s = lambda s: kiwi.join(t for t in kiwi.tokenize(s) if not t.tag.startswith(('S','W'))) #특수문자,웹 제거

def extract_bracket_content(text): #대괄호 내용 추출
    matches = re.findall(r'\[(.*?)\]', text)
    if matches:
        return matches[0]
    else:
        return None

def word_definition(new_word, wordtag):
  if wordtag.startswith("N"): #명사
    s_tag = "명사"
    new = chat3_sentence(new_word, s_tag)
    new = extract_bracket_content(new)
    new = remove_s(new) #특수문장,웹제거
    return new

  if wordtag.startswith("V"): #형용사,동사 모두 용언이므로 동사형으로 바꿈,
       s_tag = "동사"
       new = chat3_sentence(new_word, s_tag)
       extracted_content = extract_bracket_content(new)
       new = remove_s(extracted_content) #특수문장,웹제거
       test = kiwi.tokenize(new) #뒤에서부터 어간나올때까지 제거
       while test[-1].tag[0] != 'V':
           test.pop()
       new = kiwi.join(test, lm_search=True)
       return new
  if wordtag.startswith("MM"):
    s_tag = "관형사"
    new = chat3_sentence(new_word, s_tag)
    extracted_content = extract_bracket_content(new)
    new = remove_s(extracted_content) #특수문장,웹제거
    return new
  if wordtag.startswith("MA"):
    s_tag = "부사"
    new = chat3_sentence(new_word, s_tag)
    extracted_content = extract_bracket_content(new)
    new = remove_s(extracted_content) #특수문장,웹제거
    return new
  if wordtag.startswith("IC"):
    s_tag = "감탄사"
    new = chat3_sentence(new_word, s_tag)
    extracted_content = extract_bracket_content(new)
    new = remove_s(extracted_content) #특수문장,웹제거
    return new

@app.route('/', methods=['GET', 'POST'])
def add_cursed_word():
    try:
        # 사용자로부터 데이터 받기
        data = request.get_json()
        curse = data['curse'] #욕설
        input_word = data['input'] #사용자가 입력한 욕설
        tag = data['tag']#욕설 품사
        sentence = data['sentence'] #1차 교체문장
        
        print("Received Data:", data)
        
        kiwi.add_user_word(curse, tag)
        
        tokens = kiwi.tokenize(sentence)
        
        print("Type of sentence:", type(sentence))
        
        word_list = []
        word_list.append(curse)
        
        print(word_list)
        
        # 대체어 추출
        replace_curse = word_definition(input_word, tag) 
        
        # 파일에 쓰기
        filepath = "C:/versionTest/욕설형태소대체어.txt"
        with open(filepath, "a", encoding="utf-8") as file:
            file.write(f"\n{curse},{tag},{replace_curse}")
            
        for i, token in enumerate(tokens):
            for word in word_list:
                if token.form == word:
                    with open('C:/versionTest/욕설형태소대체어.txt', 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            dic_word, dic_form, dic_replace = line.strip().split(',')
                            if dic_word == token.form:
                                tokens[i] = (dic_replace, token.tag)
                                break
        
        complete_sentence = kiwi.join(tokens, lm_search=True)
        print(complete_sentence)
                    
        return jsonify({'message': '성공적으로 저장되었습니다.',
                        "complete_sentence": complete_sentence}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)



