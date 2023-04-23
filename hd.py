import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier, export_text



df = pd.read_csv('heart.csv')


df = pd.get_dummies(df, columns=['Smoking', 'AlcoholDrinking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer'])


df = df.replace({'Yes': 1, 'No': 0})


X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']


true_data = df[df['HeartDisease'] == 1].sample(n=4)
false_data = df[df['HeartDisease'] == 0].sample(n=4)
sample_data = pd.concat([true_data, false_data])


selected_features = random.sample(list(X.columns), 5)


sample_data = sample_data[selected_features + ['HeartDisease']]
train_data = sample_data.sample(frac=0.8)
test_data = sample_data.drop(train_data.index)


train_data['HeartDisease'] = train_data['HeartDisease'].astype(int)
test_data['HeartDisease'] = test_data['HeartDisease'].astype(int)


dtc = DecisionTreeClassifier(criterion='gini', splitter='best')
dtc.fit(train_data[selected_features], train_data['HeartDisease'])


text_representation = export_text(dtc, feature_names=selected_features)
print(text_representation)
