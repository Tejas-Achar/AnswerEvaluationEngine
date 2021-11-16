from flask import Flask, request, url_for
import flask
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from wordcloud import WordCloud, STOPWORDS
import numpy as npy
from PIL import Image
import json
import requests
import os
import nltk
from flask_cors import cross_origin, CORS


app = Flask(__name__)
CORS(app)
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
# Client_Id = "521049905005-ql6rv7c0csmni3tl9ltb01kpbk9t9lnr.apps.googleusercontent.com"
# Client_Secret = "yZSW5ILtoy9B2HNat08bNuwr"
# Authurl = "https://oauth2.googleapis.com/token"
# Refresh_token = "1//04ztLeUz3XJDDCgYIARAAGAQSNwF-L9Ir3dvZph5zVmXzAEe1tWZFdvmGPlA6K1r30KV-yBYBZjeB5qR4WrQiCJJRJPCmC2fsTno"


# def GetAuthToken():
#         headers = {'Content-type': 'application/json'}
#         data = {
#             "client_id": Client_Id,
#             "client_secret": Client_Secret,
#             "refresh_token": Refresh_token,
#             "grant_type": "refresh_token"
#         }
#         response = requests.post(Authurl, headers=headers, json=data)
#         respobj = response.json()
#         print("+++++++++++++++",respobj)
#         accesstoken = respobj["access_token"]
#         print(accesstoken)
#         return accesstoken
# access_Token = GetAuthToken()

@app.route('/',methods=["POST"])
@cross_origin(origin='*')
def GenerateWordCloudMain():
    AnswerData = request.get_json()
    spell = SpellChecker()
    Lem = WordNetLemmatizer()
    testString = AnswerData["student"]
    modelAnswer = AnswerData["model"]
    Question = AnswerData["question"]
    Max_Score = AnswerData["maxmarks"]
    Score_Per_word = int(AnswerData["score_per_word"])
        
    def UploadFile(filename):
        
#         headers = {
#             "Authorization": "Bearer " + access_Token}
#         para = {
#             "name": filename,
#             "parents": ["1iO5xkV3CDoEGrfvL3toFYg3tZDzj0S-z"]
#         }
#         files = {
#             'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
#             'file': open("./" + filename, "rb")
#         }
#         r = requests.post(
#             "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
#             headers=headers,
#             files=files
#         )
#         Image_Data = r.json()
#         print(r.text)
#         Image_ID = Image_Data["id"]
#         Image_Url = "https://drive.google.com/uc?export=view&id=" + str(Image_ID)
#         print(Image_Url)
        StudentScore = calculate_score(questionKeywords, modelAnswerKeywords, studentkeywords, Max_Score)
        res = flask.jsonify({"url":"https://answer-evaluation-engine.herokuapp.com"+filename,"score":StudentScore})
        #res.headers.add('Access-Control-Allow-Origin', '*')
        #res.headers.add("Access-Control-Allow-Headers", "X-Requested-With")
        return res
    def merge_images_top_bottom(file1, file2, cloud_type):
        image1 = Image.open(file1)
        image2 = Image.open(file2)
        (width1, height1) = image1.size
        (width2, height2) = image2.size
        result_width = max(width1, width2)
        result_height = height1 + height2
        result = Image.new('RGB', (result_width, result_height))
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(0, height1))
        result.save(cloud_type + ".png")
        os.remove(file1)
        os.remove(file2)
        
    def merge_images(file1, file2):
        image1 = Image.open(file1)
        image2 = Image.open(file2)
        (width1, height1) = image1.size
        (width2, height2) = image2.size
        result_width = width1 + width2
        result_height = max(height1, height2)
        result = Image.new('RGB', (result_width, result_height))
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(width1, 0))
        result.save("test.png")
        os.rename("test.png","./static/test.png")
        os.remove(file1)
        os.remove(file2)
        return UploadFile(url_for('static',filename='test.png'))

    def Process_Text(texttoprocess):
        LematizedWords = []
        testString = re.sub(r'[^\w\s]', '', texttoprocess)
        tokenwords = word_tokenize(testString)
        # print(tokenwords)
        tokenwordsw = [word for word in tokenwords if not word in stopwords.words()]
        misspelled = spell.unknown(tokenwordsw)
        # print(tokenwordsw)
        for word in misspelled:
            # print(spell.correction(word))
            for i, j in enumerate(tokenwordsw):
                if j == word:
                    tokenwordsw[i] = list(spell.candidates(word))[0]
            # Get a list of `likely` options
            # print(spell.candidates(word))
        # print(tokenwordsw)
        for i in tokenwordsw:
            LematizedWords.append(Lem.lemmatize(i))
        # print(LematizedWords)
        return (LematizedWords)
    studentAnswerKeywords = Process_Text(testString)
    modelAnswerKeywords = Process_Text(modelAnswer)
    questionKeywords = Process_Text(Question)
    uniqueKeywords = list(set(modelAnswerKeywords) - set(questionKeywords))
    # print(modelAnswerKeywords)
    # print(studentAnswerKeywords)
    def create_word_cloud(Answer, AnswerType, color, image_type):
        # Use cloud image mask to outline words
        maskArray = npy.array(Image.open(image_type))
        # configure cloud
        cloud = WordCloud(background_color="white", color_func=lambda *args, **kwargs: color, max_words=200,
                          mask=maskArray, stopwords=set(STOPWORDS))
        # generate cloud from input string
        cloud.generate(Answer)
        # save file as .png image
        cloud.to_file(AnswerType + ".png")
    def syn_gen(AnswerWord):
        syn = []
        syn_in_model = []
        for synset in wordnet.synsets(AnswerWord):
            for lemma in synset.lemmas():
                syn.append(lemma.name())  # add the synonyms
                if lemma.name() in modelAnswerKeywords:
                    syn_in_model.append(lemma.name())
        # print('Synonyms: ' + str(syn))
        return AnswerWord, list(set(syn))
    def ant_gen(AnswerWord):
        ant = []
        for synset in wordnet.synsets(AnswerWord):
            for lemma in synset.lemmas():
                if lemma.antonyms():  # When antonyms are available, add them into the list
                    ant.append(lemma.antonyms()[0].name())
        print('Antonyms: ' + str(ant))
        return ant
    def syn_checker(AnswerList):
        synlist = []
        StudentAnswerSynonyms = []
        Original_word = []
        for i in AnswerList:
            Original_word.append(syn_gen(i)[0])
            synlist.append(syn_gen(i)[1])
        for i in synlist:
            for j in i:
                # print(j)
                if j in uniqueKeywords:
                    print(j)
                    StudentAnswerSynonyms.append(str(j))
        return StudentAnswerSynonyms
    def ant_checker(AnswerList):
        antlist = []
        StudentAnswerAntonyms = []
        for i in AnswerList:
            antlist.append(ant_gen(i))
        for i in antlist:
            # print(i)
            for j in i:
                # print(j)
                if j in modelAnswerKeywords:
                    print(j)
                    StudentAnswerAntonyms.append(j)
        return StudentAnswerAntonyms
    def calculate_score(question, modelanswer, studentanswer, maxscore):
        uniquekeys = list(set(modelanswer) - set(question))
        print("Unique Keywords : ", uniquekeys)
        #scoreperkeyword = maxscore / len(uniquekeys)
        scoreperkeyword = Score_Per_word
        if scoreperkeyword <= 0.1:
            scoreperkeyword = 0.1
        print("Score Per Keyword : ", scoreperkeyword)
        studentkeywords = list(set(uniquekeys).intersection(set(studentanswer)))
        print("student Answer : ", studentkeywords)
        studentscore = len(studentkeywords) * scoreperkeyword
        if studentscore >= maxscore:
            studentscore = maxscore
        print("Student Score : ", studentscore)
        return format(studentscore,".1f")

    StudentAnswerSynonyms = syn_checker(studentAnswerKeywords)
    print(StudentAnswerSynonyms)
    studentkeywords = list(set(uniqueKeywords).intersection(set(studentAnswerKeywords)))
    studentkeywords = list(set(studentkeywords + StudentAnswerSynonyms))
    finalStudentAnswerKeywords = ' '.join([str(elem) for elem in studentkeywords])
    # modelAnswerKeywords = ' '.join([str(elem) for elem in modelAnswerKeywords])
    extraWordsStudentAnswer = list(set(studentAnswerKeywords) - set(modelAnswerKeywords))
    extraWordsModelAnswer = list(set(modelAnswerKeywords) - set(studentAnswerKeywords))
    extraWordsStudentAnswer = ' '.join([str(elem) for elem in extraWordsStudentAnswer])
    extraWordsModelAnswer = ' '.join([str(elem) for elem in extraWordsModelAnswer])
    commonModelAnswerKeywords = list(set(modelAnswerKeywords) - set(extraWordsModelAnswer))
    commonModelAnswerKeywords = ' '.join([str(elem) for elem in commonModelAnswerKeywords])
    extraWords = extraWordsStudentAnswer + extraWordsModelAnswer
    print(studentAnswerKeywords)
    print(modelAnswerKeywords)
    print(str(extraWords))
    extraWords = ' '.join([str(elem) for elem in extraWords])
    # generate unCommon keywords
    if len(extraWordsStudentAnswer) != 0:
        create_word_cloud(str(extraWordsStudentAnswer), "topUncommon", "blue", "cloud_top.PNG")
    else:
        create_word_cloud("No_Extra_words_in_Student_Answer", "topUncommon", "blue", "cloud_top.PNG")
    if len(extraWordsModelAnswer) != 0:
        create_word_cloud(str(extraWordsModelAnswer), "bottomUncommon", "purple", "cloud_bottom.PNG")
    else:
        create_word_cloud("No_Extra_words_in_Model_answer", "bottomUncommon", "purple", "cloud_bottom.PNG")
    merge_images_top_bottom("topUncommon.png", "bottomUncommon.png", "uncommon")
    # generate Common keywords
    if len(finalStudentAnswerKeywords) != 0 and str(finalStudentAnswerKeywords) !="The":
        create_word_cloud(str(finalStudentAnswerKeywords), "common", "red", "cloud.PNG")
    else:
        create_word_cloud("No_Common_Keywords_In_model_And_Student_Answer", "common", "red", "cloud.PNG")
   
    return merge_images("common.png", "uncommon.png")
    
if __name__ == '_main_':
    app.run()
