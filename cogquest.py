from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from flask import Flask, jsonify,request
from bson.json_util import dumps
import numpy as np
import math
import joblib
import cv2
import requests

app=Flask(__name__)

connection_string = r'mongodb://aromongo:%40r0dek%40412@mongodb.arodek.com:27017/?authMechanism=DEFAULT'
database_name = r'Cogniquest'
collection_name_ica=r'Ica_Test'
result_=r'Result_Display'
test_history=r'TestHistory'

client = MongoClient(connection_string)
db = client[database_name]
collection_ica = db[collection_name_ica]
collection_result = db[result_]
collection_test_and_user_id=db[test_history]
loaded_model = joblib.load('linear_regression_model.pkl')

#Naming Test(Q3)
def naming_test(answer_new):
    value = [string.lower() for string in answer_new]
    list_1=[]
    list_2=[]
    list_3=[]
    j = 0
    result_dict={}
    text = ["lion", "camel", ["rhinoceros", "rhino"]]
 
    for index, i in enumerate(value):
        if isinstance(text[index], list):
            if i in text[index]:
                list_1.append(i)
                list_2.append(index)
            else:
                list_3.append("not in list")
        else:
            if i == text[index]:
                list_1.append(i)
                list_2.append(index)
            else:
                list_3.append("not in list")
    try:
        for i in range(len(list_2)):
            result_dict[list_2[i]] = list_1[i]
        for i in result_dict.keys():
            if isinstance(text[i], list):  
                if value[i] in text[i]:
                    j += 1
            elif value[i] == text[i]:
                j += 1
        score_3 = j
        return score_3
    except TypeError:
        return 0 

#Attention Test(Q4)
def attention_test(answer_new):
    value = [answer_new[key] for key in answer_new]
    j=0
    text=['2159836','253407']
    for i,(val, txt) in enumerate(zip(value, text)):
        if val == txt:
            j += 1
    score_4=j
    return score_4

#Language Test(Q5)
def language_test(answer_new):
    value_answer=answer_new.split(',' ' ')
    value = [word.lower() for word in value_answer]
    text = ['cab', 'can', 'cub', 'cot', 'cow', 'cry', 'care', 'crow', 'chair', 'charm', 'chore', 'choir', 'chamber', 'charity', 'clove', 'cloud', 'centre', 'convent', 'concern', 'covenant', 'caricature', 'character', 'courage', 'counterpart', 'catch', 'cover', 'clone', 'cut', 'cast', 'crave', 'cite', 'cede', 'climb', 'close', 'chirp', 'colour','come', 'cave', 'cheer', 'count', 'crack', 'certify', 'comfort', 'crumble', 'challenge', 'characterise', 'cute', 'calm', 'clean', 'correct', 'cunning', 'conducive', 'courageous', 'charitable', 'canned', 'careful', 'careless', 'carefree', 'crumbled', 'closed', 'crunchy', 'creepy', 'critical', 'covered', 'colourful', 'concerned', 'chapped', 'clouded', 'cheerful', 'call', 'class', 'clutter', 'chatter', 'classy', 'cone', 'case', 'cupboard', 'conceive', 'cubicle', 'clad', 'clueless', 'cobweb', 'cope', 'cease', 'cleft', 'cracker', 'cough', 'cost', 'chandelier', 'cat', 'camel', 'coupon', 'clear', 'cloudy', 'caring', 'creative', 'clumsy', 'comfortable', 'clock', 'computer', 'cap', 'candy', 'cotton', 'captain', 'camera', 'coal', 'cucumber', 'cottage', 'chalk', 'car', 'curd', 'cart', 'card', 'cabin', 'cabinet', 'cock', 'cake', 'cashew', 'chocolate', 'comb', 'candle', 'crocodile', 'cross', 'christmas', 'cluster', 'cup', 'coin']
    common_words = set(value) & set(text)
    if len(common_words)>=11:
        score_5=1
    else:
        score_5=0
    return score_5

#Abstraction Test(Q6)
def abstraction_test(answer_new):
    j=0
    value = answer_new.lower()
    new_value=[letter.replace(" " ,"_") for letter in value]
    my_lst_str = ''.join(map(str, new_value))
    text = ["vehicle","vehicle_","used_for_transportation","transport_","transport_vehicle"]

    if my_lst_str in [item for item in text]:
        j +=1
    else:
        j=j

    if j==1:
        score6=1
    else:
        score6=0
    return score6

#Delayed Recall Test(Q7)
def delayed_recall_test(answer_new):
    value = [string.lower().strip() for string in answer_new]
    list_1 = []
    list_2 = []
    list_3 = []
    j = 0
    result_dict = {}
    text = ["banana", "milk", "deer"]

    for index, i in enumerate(value):
        if i in text:
            list_1.append(i)
            list_2.append(index)
        else:
            list_3.append("not in list")

    try:
        for i in range(len(list_2)):
            result_dict[list_2[i]] = list_1[i]
        for i in result_dict.keys():
            if value[i] == text[i]:
                j += 1
        score7 = j
        return score7
    except TypeError:
        return 0

def get_result(user_id:str,test_id:str):
    global connection_string,database_name
    info_dict={'collection_name_3': r'qcollection2',
                'collection_name_4': r'qcollection4',
                'collection_name_5':r'qcollection5',
                'collection_name_6': r'qcollection6',
                'collection_name_7': r'qcollection7',
                'collection_name_1': r'qcollection1',
                'user':r'users',
                'Result':r'Result_Display',
                'userId':user_id,
                'testId':test_id}
    keys_to_exclude = ['userId', 'user','Result','testId','collection_name_1']
    original_score=0
    filtered_values = [value for key, value in info_dict.items() if key not in keys_to_exclude]
    object_id_user=info_dict['userId']
    object_id=info_dict['testId']
    client = MongoClient(connection_string)  
    db = client.get_database(database_name)
    age = db[info_dict['user']].find_one({'_id': object_id_user})['age']
    #end_time=db[info_dict['collection_name_7']].find_one({"testId": object_id})['testTime']
    #score_6=image_intensity(testid,db[info_dict['collection_name_1']])
    #score calculation
    collection_name = [db[collection] for collection in filtered_values]
    for i in collection_name:
        try:
            result_test =i.find_one({'testId':object_id})['testData']
            try:
                score_1=delayed_recall_test(result_test)
            except:
                score_1=0
            try:
                score_2=abstraction_test(result_test)
            except:
                score_2=0
            try:
                score_3=language_test(result_test)
            except:
                score_3=0
            try:
                score_4=attention_test(result_test)
            except:
                score_4=0
            try:
                score_5=naming_test(result_test)
            except:
                score_5=0
            overall_score=max(score_1,score_2,score_3,score_4,score_5)
            original_score=original_score+overall_score
        except:
            print("not find in database")
    user_report={ 
    'age':age,
    'MOCA_Score':original_score
    }
    return user_report

def ica_test(answer_new):
    value = [string.lower() for string in answer_new]
    text = ["animal", "non-animal"]
    count1 = 0
    count2 = 0
    count3 = 0
    total_animal_count=5
    for i in value:
        if i in text[0]:
            count1 += 1
        elif i in text[1]:
            count2 += 1
        else:
            count3=count3
    if count1==5:
        accuracy=(count1/total_animal_count)*100
    else:
        accuracy=0
    return accuracy

def Ica_score(user_id:str,test_id:str):
    try:
        user_report={ 
        'user_id':user_id,
        'test_id':test_id,
        'age':0,
        'MOCA_Score':0,
        'Accuracy':0,
        'Speed':0,
        'ICA_Index':0,
        'ICA_Score':0,
        'date':datetime.now().strftime("%Y-%m-%d"),
        'timestamp':datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f %z")
        }
        try:
            test_data = collection_ica.find_one({'testId':user_report['test_id']})['testData']
            test_time = collection_ica.find_one({'userId':user_report['user_id']})['testTime']
            speed_ica_text = round(min(100, 100 * (1 / math.exp((int(str(test_time).split('.')[0]) * 60 + int(str(test_time).split('.')[1])) / 1025))), 2)
            user_report['Speed']=speed_ica_text
            user_report['Accuracy']=ica_test(test_data)
            user_report['ICA_Index']=round((((user_report['Speed'])/100*(user_report['Accuracy'])/100)*100),2)
            #print(user_report)
            if user_report['ICA_Index']>=50:
               user_report['MOCA_Score']= get_result(user_id,user_report['test_id'])['MOCA_Score']
               user_report['age']= get_result(user_id,user_report['test_id'])['age']
               test_data = [user_report['age'], user_report['ICA_Index'], user_report['MOCA_Score']]
               new_input = np.array([test_data])
               predicted_probability = loaded_model.predict(new_input.reshape(1, -1))
               user_report['ICA_Score'] = round(abs((1-predicted_probability[0]))*100,2)
               collection_result.insert_one(user_report)
            else:
               user_report['MOCA_Score']= 0
               user_report['age']= 0
               user_report['ICA_Score'] = 0
               collection_result.insert_one(user_report)
            return dumps(user_report)
        except Exception as e:
            user_report['Speed']=0
            user_report['Accuracy']=0
            user_report['ICA_Index']=0
            collection_result.insert_one(user_report)
            return dumps(user_report)
    except Exception as e:
        return dumps({"error": 'data_missing'})

def get_user_id_and_test_id(user_id:str,test_id:str):
    test_id_store=ObjectId(test_id)
    user_id=collection_test_and_user_id.find_one({'_id':ObjectId(test_id_store)})['userId']
    user_id_store=ObjectId(user_id)
    try:
        return Ica_score(user_id_store,test_id_store)
    except:
        return dumps({'data':"missing"})


@app.route('/<string:user_id>/<string:test_id>', methods=['GET'])
def get_result_by_test_id(user_id,test_id):
    list_inputs =[user_id,test_id]
    if user_id and test_id not in list_inputs:
        return jsonify({'error': '404 User not found'})
    else:
        try:
            return get_user_id_and_test_id(user_id,test_id)
        except TypeError as e:
            return jsonify({"error": e})
        except KeyError :
            return jsonify({"error": 'key_error'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')   
    
    