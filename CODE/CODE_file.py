from sklearn.model_selection import train_test_split
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
dt: object=pd.read_csv("re/car.data")
print(dt)
X=dt[["buying","maint","safety"]].values
y=dt[["class"]]
print(X,y)
gb=dt[["class","buying"]]
gb=gb.set_index("class")
gb=gb.groupby("class")
gb=gb.count()
gb.plot(kind="bar")
plt.show()
print(gb)
#string to num
ln=LabelEncoder()
for i in range(len(X[0])):
    X[:,i]=ln.fit_transform(X[:,i])

print(X)
lm={
    "unacc":0,
    'acc':1,
    'good':2,
    'vgood':3
}
y["class"]=y["class"].map(lm)
y=np.array(y)
print(y)
x_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_test)
model=SVC()
model.fit(x_train,y_train)
predi=model.predict(X_test)
acc=accuracy_score(y_test,predi)
print(acc)
confusion=confusion_matrix(y_test,predi)
print(confusion)
sn.heatmap(confusion,annot=True,xticklabels=["unacc",'acc','good','vgood'],yticklabels=["unacc",'acc','good','vgood'])
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.show()

t=np.array([[0,1,2]])
print(t)
re=model.predict(t)
print(re)
if re[0]==0:
    print("unacc")
if re[0]==1:
    print("acc")
if re[0]==2:
    print("good")
if re[0]==3:
    print("vgood")

