import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork():
    def __init__(self, ):
        self.inputSize = 34
        self.outputSize = 1
        self.hiddenSize = 34

        self.W1 = np.random.rand(self.inputSize, self.hiddenSize)
        self.W2 = np.random.rand(self.hiddenSize, self.outputSize)
        self.limit = 0.5

        self.error_list = []
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def forward(self, X):
        self.z = np.matmul(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = o - y
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = np.matmul(self.o_delta, np.matrix.transpose(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += np.matmul(np.matrix.transpose(X), self.z2_delta)
        self.W2 += np.matmul(np.matrix.transpose(self.z2), self.o_delta)

    def train(self, X, y, epochs):
        y = np.array([[i] for i in y])

        for epoch in range(epochs):
            o = self.forward(X)
            j = (-1/y.shape[0]) * np.sum(np.multiply(y, np.log(o)) + np.multiply((1 - y), np.log(1 - o)))
            print(j)
            self.backward(X, y, o)
            self.error_list.append(np.abs(self.o_error).mean())

    def predict(self, x_predicted):
        return self.forward(x_predicted).item()

    def view_error_development(self):
        plt.plot(range(len(self.error_list)), self.error_list)
        plt.title('Mean Sum Squared Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def test_evaluation(self, input_test, output_test):
        # for i, test_element in enumerate(input_test):
        #     if self.predict(test_element) > self.limit and output_test[i] == 1:
        #         self.true_positives += 1
        #     if self.predict(test_element) < self.limit and output_test[i] == 1:
        #         self.false_negatives += 1
        #     if self.predict(test_element) > self.limit and output_test[i] == 0:
        #         self.false_positives += 1
        #     if self.predict(test_element) < self.limit and output_test[i] == 0:
        #         self.true_negatives += 1
        #
        #     # print(f'{self.predict(test_element)} ---- {output_test[i]}')
        # print('True positives: ', self.true_positives, '\nTrue negatives: ', self.true_negatives,
        #       '\nFalse positives: ', self.false_positives, '\nFalse negatives: ', self.false_negatives,
        #       '\nAccuracy: ', (self.true_positives + self.true_negatives) /
        #       (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives))
        print('----- MANUAL TESTING -----')
        sum = 0
        preds = []
        for test_element in input_test:
            pred = self.predict(test_element)
            preds.append(pred)

        for i in range(len(preds)):
            if preds[i] == output_test[i]:
                sum += 1

        print(preds)
        print(output_test)

        print(f'{sum} / {len(preds)}')
