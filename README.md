# ML-data-analysis
A deep analysis and prediction of the data based on breast cancer classification..

----------------------------------------------------------------------------------------------------------

#work flow of sci-kit learn
'''
1. setting up the data
2. creating the model
3. training the model
4. testing the model
5. prediction
'''
#importing important dependencies
import pandas as pd
import numpy as np
#importing important dependencies
import pandas as pd
import numpy as np
# Tumors in Brest Cancer
'''
1. Benign _treatable Tumor
2. Malignant - non treatable
'''
#importing dataset
import sklearn.datasets
#importing accuracy score function
from sklearn.metrics import accuracy_score
#import Train Test split function
from sklearn.model_selection import train_test_split
#importing the dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
breast_cancer_dataset
#load the dataset into dataframe
df= pd.DataFrame(breast_cancer_dataset.data, columns =breast_cancer_dataset.feature_names)
df
#checking the distribution of target (label) column
#1- Benign
#0- Malignant
df['Label'].value_counts()
#Group by label columns and find the mean for the all the columns
df.groupby(by= 'Label').mean()
#input feature (All the columns expexct the label)-x
# target column - (Lable column)-Y
X = df.drop(columns='Label',axis=1)
Y = df['Label']
#splitting the data into training data and testing data
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y, test_size= 0.2,random_state=2 )

#Model training
model = LogisticRegression()
# Training the ML model with training data
model.fit(X_train ,Y_train)
#Model Evalulation
x_train_prediction = model.predict(X_train)
traning_data_accuracy= accuracy_score(Y_train , x_train_prediction)
print("Accuracy score for the training data ",traning_data_accuracy)
#Model Evalulation
x_test_prediction = model.predict(X_test)
testing_data_accuracy= accuracy_score(Y_test , x_test_prediction)
print("Accuracy score for the testning data ",testing_data_accuracy)
#Bulding a Predictive system
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259
)
input_data_as_numpy_array = np.asarray(input_data)
#Reshape the numpuy array as we are predicting for one data point
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

# Prediction of one data point
prediction = model.predict(input_data_reshape)
prediction
if(prediction==0):
 print("Breast cancer is Malignant")
else:
 print("Breast cancer is Benign")
