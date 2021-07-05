from dataset import get_x_y
from graph_plotting import performance_evaluation
from regression import RegressionModels

if __name__ == '__main__':
    r = RegressionModels()
    r.reset_parameters(get_x_y(-1))
    k1= r.multiple_regression_model()
    k2 =r.polynomial_regression()
    k3=r.decision_tree()
    k4=r.random_forest()
    r.reset_parameters(get_x_y(1,-1))
    k5=r.support_vector_regression()
    performance_evaluation(k1,k2,k3,k4,k5)






