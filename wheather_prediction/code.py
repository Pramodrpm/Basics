import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.naive_bayes import GaussianNB
df=pd.read_csv(r"C:\\Users\\Admin\\Desktop\\basics\\Basics\\Basics\\wheather_prediction\\weather.csv")
Numerics=LabelEncoder()
input=df.drop('play',axis='columns')
target=df['play']
input['outlook_n']=Numerics.fit_transform(input['outlook'])
input['temp_n']=Numerics.fit_transform(input['temper'])
input['humidity_n']=Numerics.fit_transform(input['humidity'])
input['windy_n']=Numerics.fit_transform(input['windy'])
#dropping  the string values 
inputs_n=input.drop(['outlook','temper','humidity','windy'],axis='columns')
classifier=GaussianNB()
classifier.fit(inputs_n,target)
classifier.score(inputs_n,target)
print(classifier.predict([[2,0,0,0]]))
