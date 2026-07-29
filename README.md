<H3>ENTER YOUR NAME</H3>Harini V K
<H3>ENTER YOUR REGISTER NO.</H3>212225220036
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv("/content/Churn_Modelling.csv")
df

df.isnull().sum()

#check for duplication
df.duplicated()

print(df['CreditScore'].describe())

df.info()

df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
df

Scaler=MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1

X = df1.iloc[:, :-1].values
print(X)

y = df1.iloc[:,-1].values
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```



## OUTPUT:
<img width="1897" height="643" alt="image" src="https://github.com/user-attachments/assets/90739b61-2007-4b6f-b108-0fc80dd3f39a" />

<img width="1381" height="803" alt="image" src="https://github.com/user-attachments/assets/061ad6dc-c5a5-4d94-bb1d-febe23f3b37f" />

<img width="1015" height="687" alt="image" src="https://github.com/user-attachments/assets/e217f5f5-6a11-49ac-9c20-14d4cfab17bb" />

<img width="867" height="261" alt="image" src="https://github.com/user-attachments/assets/2902f3fa-fbe7-48eb-a337-fa90d881ab19" />

<img width="477" height="277" alt="image" src="https://github.com/user-attachments/assets/ec1ca403-a7c7-494b-892a-b8e9a49988ef" />

<img width="877" height="302" alt="image" src="https://github.com/user-attachments/assets/56bec22f-6482-40cc-a196-40202f0e3867" />

<img width="612" height="327" alt="image" src="https://github.com/user-attachments/assets/73831417-3176-481f-8237-ff787d6ac23b" />

<img width="487" height="192" alt="image" src="https://github.com/user-attachments/assets/f110f5fc-bd24-4b84-8c0a-849c987955b5" />

<img width="426" height="40" alt="image" src="https://github.com/user-attachments/assets/41e8d753-138a-4bab-9f3b-c4f4c64f5e2f" />

<img width="788" height="165" alt="image" src="https://github.com/user-attachments/assets/024a1b7d-4cc5-4993-a482-c40dc65996ae" />

<img width="1011" height="193" alt="image" src="https://github.com/user-attachments/assets/73dab7b7-4913-4088-b2a0-3cbcd143b506" />






## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


