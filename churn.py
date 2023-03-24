import pandas as pd
import numpy as np

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()

# Convert categorical variables into numerical variables
from sklearn.preprocessing import LabelEncoder, StandardScaler
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges']
le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Convert Churn variable into binary labels
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Handle missing values
df = df.dropna()

X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

dt = DecisionTreeClassifier(criterion = 'entropy',random_state= 0) 
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)
dt_pred = y_pred

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test,y_pred))
print('Accuracy is ' + str(accuracy_score(y_test, y_pred)*100) + ' %.')

lr = LogisticRegression(random_state= 0) 
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
lr_pred = y_pred

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test,y_pred))
print('Accuracy is ' + str(accuracy_score(y_test, y_pred)*100) + ' %.')


rf = RandomForestClassifier(random_state= 0) 
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)
rf_pred = y_pred

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test,y_pred))
print('Accuracy is ' + str(accuracy_score(y_test, y_pred)*100) + ' %.')

gb = GradientBoostingClassifier(random_state= 0) 
gb.fit(X_train,y_train)


from sklearn.ensemble import StackingClassifier

# define the base models
base_models = [('dt', DecisionTreeClassifier(criterion = 'entropy',random_state= 0)),
               ('lr', LogisticRegression(random_state= 0)),
               ('rf', RandomForestClassifier(random_state= 0)),
               ('gb', GradientBoostingClassifier(random_state= 0))]

# define the meta-classifier
meta_model = LogisticRegression(random_state= 0)

# create the stacked ensemble model
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# fit the model on the training data
stacked_model.fit(X_train, y_train)

# make predictions on the test data
y_pred = stacked_model.predict(X_test)

# evaluate the model performance
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test,y_pred))
print('Accuracy is ' + str(accuracy_score(y_test, y_pred)*100) + ' %.')
y_pred = gb.predict(X_test)
gb_pred = y_pred

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test,y_pred))
print('Accuracy is ' + str(accuracy_score(y_test, y_pred)*100) + ' %.')