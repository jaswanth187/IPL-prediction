import pickle
import numpy as np
# file='model.pkl'
# fileobj=open(file,'rb')
# pred=pickle.load(fileobj)
# print(pred)

with open('model.pkl','rb') as f:
    model=pickle.load(f)

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('inv_vocab.pkl','rb') as f:
    inv_voacb=pickle.load(f)

    team1=inv_voacb['Royal Challengers Bangalore']
    team2=inv_voacb['Sunrisers Hyderabad']
arr=np.array([team1,team2,int('1'),int('0')]).reshape(1,-1)
predict =model.predict(arr) 

print(predict)