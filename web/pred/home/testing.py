import pickle as pkl 
import numpy as np

with open('vocab.pkl', 'rb') as f:
            vocab = pkl.load(f)

with open('inv_vocab.pkl', 'rb') as f:
            inv_vocab = pkl.load(f)

with open('model.pkl', 'rb') as f:
    model = pkl.load(f)

team1=vocab['Royal Challengers Bangalore']
team2=vocab['Sunrisers Hyderabad']
lst = np.array([team1,team2, 0, 0], dtype='int32').reshape(1,-1)
prediction = model.predict(lst)
print(prediction)
        