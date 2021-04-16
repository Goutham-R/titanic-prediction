import pandas as pd
import seaborn as sns
import numpy
import matplotlib.pyplot as plt
def cabin_replace(dt):
    dt["Cabin"]=dt["Cabin"].fillna("Z")
    dt["Cabin"]=dt["Cabin"].apply(lambda x: str(x)[0])
    return dt    
def title(dt):
    dt["Title"]=dt["Name"].apply(lambda x: str(x).split(",")[1].split(".")[0])
    return dt
def replace_title(dt):
    dt["Title"].replace(to_replace=[" Capt"," Don"," Jonkheer"," Lady"," Mme"," Ms"," Sir"," the Countess"," Rev"],value="Rare",inplace=True)
    dt["Title"].replace(to_replace=[" Col"," Major"," Mlle"],value="Medium",inplace=True)
    dt["Title"].replace(to_replace=[" Dr"," Master"," Miss"," Mr"," Mrs"],value="High",inplace=True)
    return dt
def replace_cabin(df):
    df["Cabin"].replace(to_replace=["B","D","E"],value="H",inplace=True)
    df["Cabin"].replace(to_replace=["C","F"],value="M",inplace=True)
    df["Cabin"].replace(to_replace=["A","Z","G"],value="L",inplace=True)
    return df
def family(dt):
    dt["Family"]=dt["SibSp"]+dt["Parch"]+1
    dt["Alone"]=dt["Family"].apply(lambda x: 1 if x==1 else 0)
    dt.drop(["Parch","SibSp"],axis=1,inplace=True)
    return dt
def age_group(dt):
    dt.loc[dt["Age"]<=16,"AgeGroup"]=1
    dt.loc[(dt["Age"]>16)&(dt["Age"]<=40),"AgeGroup"]=2
    dt.loc[(dt["Age"]>40)&(dt["Age"]<=60),"AgeGroup"]=3
    dt.loc[(dt["Age"]>60),"AgeGroup"]=4
    return dt
path_train="train.csv"
path_test="test.csv"
dt_train=pd.read_csv(path_train)
dt_test=pd.read_csv(path_test)
sns.set()
#sns.distplot(dt_train["Age"],bins=50)
#plt.xlim(0,None)
#sns.boxplot(x=dt_train["Pclass"],y=dt_train["Age"])
#sns.violinplot(x=dt_train["Pclass"],y=dt_train["Age"])
#print(dt_train["Cabin"])
dt_train=cabin_replace(dt_train)
dt_test=cabin_replace(dt_test)
dt_train=replace_cabin(dt_train)
dt_test=replace_cabin(dt_test)
#sns.barplot(data=dt_train,x="Cabin",y="Survived",order=["A","B","C","D","E","F","G","Z","T"])
#plt.show()
dt_train=title(dt_train)
dt_train=replace_title(dt_train)
dt_test=title(dt_test)
dt_test=replace_title(dt_test)
#print(dt_train["Title"].values)
dt_train=family(dt_train)
#sns.barplot(x=dt_train["Alone"],y=dt_train["Survived"])
#plt.show()
dt_test=family(dt_test)
#g=sns.distplot(dt_train["Age"])
#plt.show()
dt_train=age_group(dt_train)
dt_test=age_group(dt_test)
#sns.barplot(data=dt_train,x="AgeGroup",y="Survived")
#plt.show()
#sns.distplot(dt_train["Fare"])
#plt.xlim(0,200)
#plt.show()
#sns.heatmap(dt_train.corr(),square=True,annot=True)
#plt.show()
#print(dt_train.columns)
dt_train["Sex"]=dt_train["Sex"].map({"male":0,"female":1})
dt_train["Embarked"]=pd.get_dummies(dt_train["Embarked"])
dt_train["Age"]=dt_train["Age"].fillna(dt_train["Age"].mean())
dt_test["Sex"]=dt_test["Sex"].map({"male":0,"female":1})
dt_test["Embarked"]=pd.get_dummies(dt_test["Embarked"])
dt_test["Age"]=dt_test["Age"].fillna(dt_test["Age"].mean())
x=dt_train.drop(["Ticket","Title","Name","Survived","Cabin","Fare","Family","Alone","AgeGroup"],axis=1)
y=dt_train["Survived"]
test_pre=dt_test.drop(["Ticket","Title","Name","Cabin","Fare","Family","Alone","AgeGroup"],axis=1)
#print(dt_train["Sex"])
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(x)
x_array=np.array(x).copy()
y_array=np.array(y).copy()
models=[RandomForestClassifier(),AdaBoostClassifier(),BaggingClassifier(),ExtraTreesClassifier(),SVC(),KNeighborsClassifier()]
names=["RandomForestClassifier()","AdaBoostClassifier()","BaggingClassifier()","ExtraTreesClassifier()","SVC()","KNeighborsClassifier()"]
scores=[]
#for model in models:
#    score=cross_val_score(model,X=x_array,y=y_array,cv=10,scoring="accuracy").mean()
#    scores.append(score)    
#result=pd.DataFrame(scores,index=names,columns=["Scores"]).sort_values(by="Scores",ascending=False)
#print(result)
#continuing with SVC
from sklearn.pipeline import make_pipeline
print(cross_val_score(SVC(),X=x_array,y=y_array,cv=10,scoring="accuracy").mean())
hyperparameters={"kernel":["linear","rbf"],"gamma":[0.1,1,10,"auto"],"C":[0.1,1,10]}
gs=GridSearchCV(SVC(),param_grid=hyperparameters,cv=5)
gs.fit(X=x_array,y=y_array)
print(gs.refit)
#clf is the model to be used
test=np.array(test_pre).copy()
test=StandardScaler().fit_transform(test)
ans=gs.predict(test)
#print(ans)
dicti={"PassengerId":dt_test["PassengerId"],"Survived":ans}
dicti=pd.DataFrame(dicti)
dicti.to_csv("Ouput2.csv",index=False)
