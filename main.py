from simplennet.neural_network import NeuralNetwork
from subprocess import CalledProcessError
import numpy as np
import subprocess
import os
import sys
from sklearn.neural_network import MLPClassifier

if not os.path.exists(os.path.join('.', 'data')):
    os.mkdir(os.path.join('.', 'data'))
    try:
        subprocess.check_call([sys.executable, os.path.join('simplennet', 'prepare_data.py')])
    except CalledProcessError:
        print('Could not load data. Please try again')
        exit(1)

x_train = np.load(os.path.join('data', 'x_train.npy'))
x_cv = np.load(os.path.join('data', 'x_cv.npy'))
x_test = np.load(os.path.join('data', 'x_test.npy'))
y_train = np.load(os.path.join('data', 'y_train.npy'))
y_cv = np.load(os.path.join('data', 'y_cv.npy'))
y_test = np.load(os.path.join('data', 'y_test.npy'))

NN = NeuralNetwork()

NN.train(x_train, y_train, 1)
print(x_test[0])
print(x_test[0].shape)
print(y_test[0])
input_pred = np.array(x_test[0])
print(NN.predict(input_pred))
NN.view_error_development()
NN.test_evaluation(x_test, y_test)

clf = MLPClassifier(
    activation='relu',
    solver='sgd',
    alpha=1e-5,
    hidden_layer_sizes=(34),
    random_state=1,
    max_iter=5000
)
clf.fit(x_train, y_train)

print( 'accuracy: ',clf.score( x_test, y_test ))

pred = clf.predict(x_test)
print(pred)
print(y_test)
sum = 0
for i in range(len(pred)):
    if pred[i] == y_test[i]:
        sum += 1

print(f'{sum} / {len(pred)}')