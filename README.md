# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Revathi.D
### Register Number:212221240045
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('dldata').sheet1


rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})

df.head()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
x=[]
y=[]
for i in range(60):
  num = i+1
  x.append(num)
  y.append(num*12) 
df=pd.DataFrame({'INPUT': x, 'OUTPUT': y})
df.head()

inp=df[["INPUT"]].values
out=df[["OUTPUT"]].values
Input_train,Input_test,Output_train,Output_test=train_test_split(inp,out,test_size=0.33)
Scaler=MinMaxScaler()
Scaler.fit(Input_train)
Scaler.fit(Input_test)
Input_train=Scaler.transform(Input_train)
Input_test=Scaler.transform(Input_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([Dense(5,activation='relu'),
                  Dense(10,activation='relu'),
                  Dense(1)])
model.compile(loss="mse",optimizer="rmsprop")
history=model.fit(Input_train,Output_train, epochs=1000,batch_size=32)

prediction_test=int(input("Enter the value to predict:"))
preds=model.predict(Scaler.transform([[prediction_test]]))
print("The prediction for the given input "+str(prediction_test)+" is:"+str(int(np.round(preds))))

model.evaluate(Input_test,Output_test)

import matplotlib.pyplot as plt
plt.suptitle("   Harish Ravishankar")
plt.title("Error VS Iteration")
plt.ylabel('MSE')
plt.xlabel('Iteration')
plt.plot(pd.DataFrame(history.history))
plt.legend(['train'] )
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default

worksheet = gc.open('dldata').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'INPUT':'float'})
dataset1 = dataset1.astype({'OUTPUT':'float'})

dataset1.head()

X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)


ai_brain = Sequential([
    Dense(3,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=X_train1,y=y_train,epochs=50)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()



```
## Dataset Information

![image](https://github.com/Revathi-Dayalan/basic-nn-model/assets/96000574/476d4b5e-c065-48e4-8e3b-3939cfd2a851)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Revathi-Dayalan/basic-nn-model/assets/96000574/c36c09b7-b8a1-454f-8090-7bfec62eda44)



### Test Data Root Mean Squared Error

![image](https://github.com/Revathi-Dayalan/basic-nn-model/assets/96000574/79495b03-a7e5-4546-a021-e166cba4ae36)


### New Sample Data Prediction

![image](https://github.com/Revathi-Dayalan/basic-nn-model/assets/96000574/fed93548-c1f5-4bf1-be5e-351d99367b2b)


## RESULT
Henceforth, a basic neural regression model has been implemented
