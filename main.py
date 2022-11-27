from simplennet.neural_network import NeuralNetwork
from subprocess import CalledProcessError
import numpy as np
import subprocess
import os
import sys

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

NN.train(x_train, y_train, 10)
input_pred = np.array([1,1,2,2,3,4404,0,4,3,1])
print(NN.predict(input_pred))
NN.view_error_development()
# NN.test_evaluation(input_test_scaled, output_test_scaled)