import pandas as pd
from sklearn.model_selection import train_test_split
dataset = pd.read_csv("Data.csv")


def get_x_y(split,split1=None):
    x=dataset.iloc[:,:split].values
    y=dataset.iloc[:,split].values
    if split1 is not None:#for svr regression
        x=dataset.iloc[:,:split:split1].values
        y=dataset.iloc[:,split1].values
        y= y.reshape(len(y),1)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    return x,y,x_train,x_test,y_train,y_test
