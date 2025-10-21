from flask import Flask, request, jsonify
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
from flask_cors import CORS
##욕설 형태소 추출 및 교체문장 생성
import os
import re
import sys
import urllib.request
import json
from kiwipiepy import Kiwi
from transformers import BertForSequenceClassification, AutoTokenizer, TextClassificationPipeline
kiwi = Kiwi()
import openai

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

#욕설 형태소 추가
with open("C:/versionTest/욕설형태소대체어.txt", "r", encoding="utf-8") as file:
    for line_num, line in enumerate(file, start=1):
        word, pos, replacement = line.strip().split(",")
        
        # Ensure no whitespace in 'word' and 'replacement'
        if " " in word or " " in replacement:
            print(f"Skipping line {line_num}: Whitespace found in word or replacement.")
            continue  # Skip to next iteration
        
        try:
            kiwi.add_user_word(word, pos)  # 욕설 단어 추가
            kiwi.add_user_word(replacement, pos)  # 대체어 추가
        except Exception as e:
            print(f"Error at line {line_num}: {str(e)}")


# 욕설 예측 함수
def predict_sentence(sentence):
    results = pipe(sentence)[0]
    for result in results:
        if result['label'] == '악플/욕설':
            return result['score']
    return 0.0  # '악플/욕설' 라벨을 찾지 못한 경우 0 반환


def predict_curse_sentences(sentences):
    curse_sentences = []
    for sentence in sentences:
        prediction = predict_sentence(sentence)  
        if prediction >= 0.5:  
            curse_sentences.append(sentence)
    return curse_sentences

def split_sentence_into_phrases(sentence):
    phrases = sentence.split(' ')  
    return phrases

def check_curse_sentences(input_sentences):
    sentences = input_sentences.split('.')
    curse_sentences = predict_curse_sentences(sentences)

    curse_phrases = []  

    for curse_sentence in curse_sentences:
        phrases = split_sentence_into_phrases(curse_sentence)
        found_curse = False

        for num_phrases in range(1, len(phrases) + 1):
            for i in range(len(phrases) - num_phrases + 1):
                current_phrases = phrases[i:i + num_phrases]
                remaining_phrases = phrases[:i] + phrases[i + num_phrases:]
                remaining_sentence = ' '.join(remaining_phrases)

                prediction = predict_sentence(remaining_sentence)
                if prediction < 0.5:
                    curse_phrases.append(current_phrases)  
                    found_curse = True
                    break

            if found_curse:
                break

    result = []  

    if curse_phrases:
        for curse_sentence in curse_sentences:
            curse_sentence_result = []  
            for curse_phrase in curse_phrases:
                if ' '.join(curse_phrase) in curse_sentence:
                    curse_sentence_result.append(curse_phrase)  
            result.append([curse_sentence, curse_sentence_result])  
    else:
        result = [[]] * len(curse_sentences)  

    return result

def remove_ending_special_web(text):                                                                   
    return kiwi.join(t for t in kiwi.tokenize(text) if not t.tag.startswith(('E', 'S', 'W','J','XP','VC','XS'))) #어미,특수문자,웹,조사,접미사등 문장변환시 어법을 방해하는 품사 제거

#데체어 없는 욕설 찾는 함수
def find_unlisted_words(word_list):
    # 딕셔너리 생성
    replace_dict = {}
    with open('C:/versionTest/욕설형태소대체어.txt', 'r', encoding='utf-8') as file:
        for line in file:
            original, _, _ = line.strip().split(',')
            replace_dict[original] = True
    
    # 리스트에서 딕셔너리에 없는 단어 찾기
    new_list = [word for word in word_list if word not in replace_dict]
    
    return new_list

#욕설의 형태소 태그 추출(질문1)
def get_pos_tag_1(sentence, tag):
    openai.api_key = "api_key"
    question = f"세종 품사 태그에 기초하여 \"{sentence}\"에서 형태소 \"{tag}\"의 대문자 영어 태그를 대괄호 안에 넣어서 출력해줘."
    #print(f"질문: {question}")  # 사용자의 질문을 출력합니다.
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

#욕설의 형태소 태그 추출(질문2)
def get_pos_tag_2(sentence, tag):
    # API 키를 여기에 입력하세요.
    openai.api_key = "api_key"
    
    question = f"세종 품사 태그에 기초하여 \"{sentence}\"에서 형태소 \"{tag}\"의 대문자 영어 태그를 대괄호 안에 넣어서 출력해줘."
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
    
    
###최종 테스트
def check_tag(sentence, tag):
  tag_list = ['NNG', 'NNP', 'NNB', 'NR', 'NP','VV','VA','VX','VCP','VCN','MM','MAG','MAJ','IC','JKS','JKC','JKG','JKO','JKB','JKV','JKQ','JX','JC','EP','EF','EC','ETN','ETM','XPN','XSN','XSV','XSA','XSM','XR']
  while True:
    answer = get_pos_tag_2(sentence, tag)
    print("답변:"+answer)
    match = re.search(r'\[(.*?)\]', answer) #대뢀호 안의 텍스트 추출
    answer_pop = match.group(1) if match else ""
    #print("대괄호 안의 결과:"+answer_pop)
    for t in tag_list: #태그리스트에 존재 여부 판단
          if t in answer_pop:
              print("태그 결과:"+t)
              return t #존재시 태그 반환



@app.route('/', methods=['POST'])
def check_curse():
    if request.method == 'POST':
        data = request.get_json()
        input_sentences = data.get('input_sentences')
        
        if not input_sentences:
            return jsonify({"error": "input_sentences is required"}), 400
        
        # 기존에 제공해주신 로직의 실행
        result = check_curse_sentences(input_sentences)
        result_replace = [[sentence, [[remove_ending_special_web(token) for token in tokens[0]]]] for sentence, tokens in result]
        word_list = []
        for item in result_replace:
            word_list.extend(item[1][0])
        tokens = kiwi.tokenize(input_sentences)
        
        
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
                            
        no_replace_list = find_unlisted_words( word_list) #대체어가 없는 욕설 리스트

        no_curse_list = [] #문장, 대체어없는 욕설, 욕설품사

        if len(no_replace_list) == 0:
            print("대체어가 있습니다.")
        else:
            print("대체어가 없습니다.")
            for sentence, curse_list in result_replace:
                for no_replace_curse in no_replace_list:
                    # 대체어가 없는 욕설이 문장에 포함되어 있는지 확인
                    if no_replace_curse in sentence:
                        # 새로운 리스트에 요소 추가
                        tag = check_tag(sentence,no_replace_curse)
                        no_curse_list.append({
                            'sentence': sentence,
                            'no_replace_curse': no_replace_curse,
                            'no_replace_curse_tag': tag
                        })
                        
        print(no_curse_list)
                            
        complete_sentence = kiwi.join(tokens, lm_search=True)
        
        # 결과를 JSON 형태로 반환
        return jsonify({
            "result": result,
            "word_list": word_list,
            "complete_sentence": complete_sentence,
            "no_curse_list": no_curse_list
        }), 200
    else:
        return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000)
