from flask import Flask, request
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

app = Flask(__name__)


nltk.download('wordnet')
nltk.download('stopwords')

@app.route('/',methods=["POST"])
def GenerateWordCloudMain():

    AnswerData = request.get_json()
    spell = SpellChecker()
    Lem = WordNetLemmatizer()

    Client_Id = "1065772476833-02d56kvkapfcce1se62m1v2vhe6bnabi.apps.googleusercontent.com"
    Client_Secret = "wZtMf8E7a1XD2WXfIT-SEWwl"
    Authurl = "https://oauth2.googleapis.com/token"
    Refresh_token = "1//04M6zq1t22YnUCgYIARAAGAQSNwF-L9Ird4Bpi4OpPigSMbum4sQ0ZWaiJ6Zp2TwPOtG7Fh1RfE3a1zfghubuAT0tnNWYlTvK1Yg"

    testString = AnswerData["student"]
    modelAnswer = AnswerData["model"]
    def GetAuthToken():
        headers = {'Content-type': 'application/json'}
        data = {
            "client_id": Client_Id,
            "client_secret": Client_Secret,
            "refresh_token": Refresh_token,
            "grant_type": "refresh_token"
        }
        response = requests.post(Authurl, headers=headers, json=data)
        respobj = response.json()
        print("+++++++++++++++",respobj)
        accesstoken = respobj["access_token"]
        print(accesstoken)
        return accesstoken

    def UploadFile(filename):
        access_token = GetAuthToken()
        headers = {
            "Authorization": "Bearer " + access_token}
        para = {
            "name": filename,
            "parents": ["1iO5xkV3CDoEGrfvL3toFYg3tZDzj0S-z"]
        }
        files = {
            'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
            'file': open("./" + filename, "rb")
        }
        r = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
            headers=headers,
            files=files
        )
        Image_Data = r.json()
        print(r.text)
        Image_ID = Image_Data["id"]
        Image_Url = "https://drive.google.com/uc?export=view&id=" + str(Image_ID)
        print(Image_Url)
        return Image_Url

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
        os.remove(file1)
        os.remove(file2)

        return UploadFile("test.png")

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

    # print(modelAnswerKeywords)
    # print(studentAnswerKeywords)

    def create_word_cloud(Answer, AnswerType):
        # Use cloud image mask to outline words
        maskArray = npy.array(Image.open("cloud.png"))
        # configure cloud
        cloud = WordCloud(background_color="white", max_words=200, mask=maskArray)
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
                if j in modelAnswerKeywords:
                    print(j)
                    StudentAnswerSynonyms.append(str(j) + "-" + i[0])
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

    StudentAnswerSynonyms = syn_checker(studentAnswerKeywords)
    print(StudentAnswerSynonyms)

    finalStudentAnswerKeywords = StudentAnswerSynonyms
    print(finalStudentAnswerKeywords)
    finalStudentAnswerKeywords = ' '.join([str(elem) for elem in finalStudentAnswerKeywords])
    # modelAnswerKeywords = ' '.join([str(elem) for elem in modelAnswerKeywords])

    extraWordsStudentAnswer = list((set(studentAnswerKeywords) - set(modelAnswerKeywords)))
    extraWordsModelAnswer = list(set(modelAnswerKeywords) - set(studentAnswerKeywords))

    extraWords = extraWordsStudentAnswer + extraWordsModelAnswer
    print(studentAnswerKeywords)
    print(modelAnswerKeywords)
    print(str(extraWords))
    extraWords = ' '.join([str(elem) for elem in extraWords])
    create_word_cloud(str(finalStudentAnswerKeywords), "StudentAnswer")
    create_word_cloud(str(extraWords), "ModelAnswer")
    # print(str(finalStudentAnswerKeywords))



    return merge_images("StudentAnswer.png", "ModelAnswer.png")

if __name__ == '_main_':
    app.run()
