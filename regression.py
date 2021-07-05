from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

class RegressionModels:

    def __init__(self):
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def multiple_regression_model(self):
        mlr = LinearRegression()
        mlr.fit(self.x_train, self.y_train)  # training the model
        y_pred = mlr.predict(self.x_test)
        np.set_printoptions(precision=2)
        k1 = r2_score(self.y_test, y_pred)
        return k1

    def polynomial_regression(self):
        poly = PolynomialFeatures(degree=4)
        poly1 = poly.fit_transform(self.x_train)
        poly_reg = LinearRegression()
        poly_reg.fit(poly1, self.y_train)  # training model
        y_pred1 = poly_reg.predict(poly.transform(self.x_test))  # testing the model
        np.set_printoptions(precision=2)
        k2 = r2_score(self.y_test, y_pred1)
        return k2

    def support_vector_regression(self):
        # feature scaling
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        self.x_train = sc_x.fit_transform(self.x_train)  # converting the inputs
        self.y_train = sc_y.fit_transform(self.y_train)
        # training SVR model
        sup_reg = SVR(kernel='rbf')
        sup_reg.fit(self.x_train, self.y_train)
        # testing and predicting
        y_pred2 = sc_y.inverse_transform(
            sup_reg.predict(sc_x.transform(self.x_test)))  # converting inputs to actual values
        k3 = r2_score(self.y_test, y_pred2)
        return k3

    def random_forest(self):
        rf = RandomForestRegressor(random_state=0, n_estimators=10)
        rf.fit(self.x_train, self.y_train)
        y_pred3 = rf.predict(self.x_test)
        k4 = r2_score(self.y_test, y_pred3)
        return k4

    def decision_tree(self):
        dt = DecisionTreeRegressor(random_state=0)
        dt.fit(self.x_train, self.y_train)
        y_pred4 = dt.predict(self.x_test)
        k5 = r2_score(self.y_test, y_pred4)
        return k5

    def reset_parameters(self,data):
        self.x,self.y,self.x_train,self.x_test,self.y_train,self.y_test = data

