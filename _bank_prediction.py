import pandas
import numpy
import matplotlib.pyplot as plt
path="D:\machine_learning\_bank_credit.csv"
names=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y"]
data=pandas.read_csv(path,names=names)
x=data.iloc[ :,:24]
y=data.iloc[ :,24]

###finding correlation

correlations=data.corr()
print(correlations)
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlations,vmax=1,vmin=-1)
plt.show()

fig.colorbar(cax)
ticks=numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

###preprocessing of dataset

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
data["class"]=le.fit_transform(data["class"])
print(data)

###training and testing of dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=1)

###knn

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print("accuracy of knn :",accuracy_score(y_test,y_pred)*100)

###naive bayes

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print("accuracy of nb:",accuracy_score(y_test,y_pred)*100)


###random forest

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print("accuracy of rf:",accuracy_score(y_test,y_pred)*100)

###decision tree

from sklearn.tree import DecisionTreeClassifier
#model=DecisionTreeClassifier()
model=DecisionTreeClassifier("entropy")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print("accuracy of dt:",accuracy_score(y_test,y_pred)*100)


###logestics

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print("accuracy of logestics:",accuracy_score(y_test,y_pred)*100)





