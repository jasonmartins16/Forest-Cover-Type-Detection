import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# read the dataset
df = pd.read_csv(r'E:\forest_cover_prediction\train.csv')

# Select top 10 numerical features
top10_numeric = [
    'Elevation', 'Slope', 'Aspect',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'
]

# separate the features and target
X = df[top10_numeric]
y = df["Cover_Type"]

# separate train-test splits
y = y - 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# train the model
model = xgb.XGBClassifier(useLabelEncoder = False, eval_metric = "mlogloss", random_state = 42)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# save the model
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(top10_numeric, "top10_features.pkl")

# evaluate
print("Classification Report \n", classification_report(y_test, model.predict(X_test)))
print(f"\nAccuracy :{accuracy_score(y_test, y_pred):.2f}","\n")
#print(f"\nConfusion Matrix :{confusion_matrix(y_test, y_pred)}")

# to plot important features
'''xgb.plot_importance(model, max_num_features = 20, importance_type = "gain", height = 0.5)
plt.title("Features")
plt.show()'''