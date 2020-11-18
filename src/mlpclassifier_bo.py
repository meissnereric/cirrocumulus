import GPy
from GPyOpt.methods import BayesianOptimization
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def f(learning_rate_init):
    learning_rate_init = np.atleast_2d(np.power(10,learning_rate_init))

    for i in range(0, learning_rate_init.shape[0]):
        # Create Classifier
        clf = MLPClassifier(random_state=1, max_iter=100,
                            activation='logistic',
                            hidden_layer_sizes=(50,),
                            solver='adam', learning_rate='constant',
                            momentum=0,
                            warm_start=False,
                            learning_rate_init=learning_rate_init[i,0]
                            )
        # Train Classifier
        clf.fit(X_train, y_train)

    # Return accuracy
    return np.atleast_2d(-clf.score(X_test, y_test))

np.random.seed(42)
X, y = make_classification(n_samples=1000, random_state=1, 
                           n_classes=2, n_clusters_per_class=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)


kernel_rbf = (GPy.kern.RBF(input_dim=1, variance=0.2, lengthscale=2.0) +
              GPy.kern.White(input_dim=1, variance=0.05))

domain = [{'name': 'Learning Rate', 'type': 'continuous', 'domain': (-5,5)}]
opt = BayesianOptimization(f=f, domain=domain, model_type='GP',
                           kernel=kernel_rbf,
                           acquisition_type = 'EI',
                           initial_design_numdata = 3,
                           )

opt.run_optimization(max_iter=30)
