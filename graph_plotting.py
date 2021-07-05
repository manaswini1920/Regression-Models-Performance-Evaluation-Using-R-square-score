import matplotlib.pyplot as plt

def performance_evaluation(k1,k2,k3,k4,k5):
    models=['multiple reg','poly reg','Decision Tree','Random Forest','SVR']
    R2_score=[k1,k2,k3,k4,k5]
    plt.bar(models,R2_score)
    plt.xlabel("R2 score")
    plt.ylim(0.91,1)
    plt.show()
