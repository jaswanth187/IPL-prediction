from django.http import response
from django.http.response import HttpResponse
from django.shortcuts import render
import pickle as pkl 
import numpy as np
# Create your views here.
def index(request):
    if request.method=="POST":
        team1=str(request.POST['list1'])
        team2=str(request.POST['list2'])
        toss_winner=int(request.POST['toss_winner'])
        fb=int(request.POST['fb'])
        print(fb,toss_winner)
        

        with open('model.pkl', 'rb') as f:
            model = pkl.load(f)
        with open('vocab.pkl', 'rb') as f:
            vocab = pkl.load(f)
        with open('inv_vocab.pkl', 'rb') as f:
            inv_vocab = pkl.load(f)

        cteam1=inv_vocab[team1]
        cteam2=inv_vocab[team2]

        # arr=np.array([team1,team2,fb,toss_winner])
        

        if team1 == team2:
            return HttpResponse('You selected the same teams')
        else:
            lst = np.array([cteam1, cteam2, fb, toss_winner], dtype='int32').reshape(1,-1)

            prediction = model.predict(lst)

            if prediction == 0:
                team_win = team1

            else:
                team_win = team2
            print(team_win)
            return HttpResponse('won'+team_win)

    return render(request,'index.html')
    # return HttpResponse('this is http')