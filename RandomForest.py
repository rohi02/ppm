import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
#creatng a dataframe
df=pd.read_csv("C://sem4finalSDP//Placement_Data_Full_Class.csv")

#data preprocessing
df['gender'].replace(['M', 'F'],[1, 0], inplace=True)
df['workex'].replace(['Yes', 'No'],[1, 0], inplace=True)
df['specialisation'].replace(['Mkt&HR','Mkt&Fin'],[1, 0], inplace=True)
df['status'].replace(['Placed','Not Placed'],[1, 0], inplace=True)
d1=pd.get_dummies(df.degree_t)
df=pd.concat([df,d1],axis="columns")
final=df.drop(['salary','Others','hsc_b','ssc_b','hsc_s','sl_no','degree_t'],axis="columns")
#------------------------------------------------------------------------------------------------

x=final[['gender','ssc_p','hsc_p','degree_p','workex','etest_p','specialisation','mba_p','Comm&Mgmt','Sci&Tech']]
y=final['status']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=69)

#-----------------------------------------------------------------------------------------------

rf=RandomForestClassifier(n_estimators=60)
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test))

with open('RandomForest','wb') as f:
    pickle.dump(rf,f)