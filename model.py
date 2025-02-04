import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns    
from sklearn.preprocessing import LabelEncoder

df=sns.load_dataset('tips')
print(df)
df.drop(['smoker'],axis=1,inplace=True)
df.drop(['sex'],axis=1,inplace=True)
le=LabelEncoder()
                                                                                                      
df['time']=le.fit_transform(df['time'])

X=df[['total_bill', 'time', 'size']]
y=df['tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

pickle.dump(model,open('model.pkl','wb'))
