import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime
import math

np.random.seed(1234)

# return list of tuples (start,end) for slicing each batch X
def get_indexes(n, batchSize):
    indexes = []  # list of (start,end) for slicing each batch X
    index = 0
    for round in range(math.ceil(n / batchSize)):
        index += batchSize
        if index > n:
            index -= batchSize
            indexes.append((index, n))
            break
        indexes.append((index - batchSize, index))
    return indexes

class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1
        self.hs = {0:np.zeros((numHidden, numInput))}
        self.yhats = {0:0}
        # TODO: IMPLEMENT ME

    def backward (self, y):
        # TODO: IMPLEMENT ME
        T = len(y)

        gradient_MSE = (self.yhats[T] - y[T-1])

        #init gradient values
        gradient_w = np.dot(gradient_MSE, np.transpose(self.hs[T]))
        gradient_U = np.zeros(self.U.shape)
        gradient_V = np.zeros(self.V.shape)


        gradient_h = np.dot(np.transpose(gradient_w), gradient_MSE)
        

        for t in np.arange(T)[::-1]:
            gradient_h = (gradient_h * (1 - self.hs[t + 1] ** 2))


            gradient_w += np.dot(gradient_MSE, np.transpose(self.hs[t])) # could be self.hs[t-1]
            gradient_U += gradient_h.dot(np.transpose(self.hs[t]))
            gradient_V += gradient_h.dot(np.transpose(self.last_x[t]))

            gradient_MSE = (self.yhats[T] - y[T-1])


            gradient_h = self.U.dot(gradient_h)

        # try wth out clip and see if it works!!!! study a little more the aspects of the algo.
        for gradint in [gradient_V, gradient_U, gradient_w]:
            np.clip(gradint, -1, 1, out=gradint)

        return gradient_V, gradient_U, gradient_w

    def forward (self, x):
        # TODO: IMPLEMENT ME

        self.last_x = x
        h_t = np.zeros((numHidden, numInput))
        yhat_t = 0

        for i, x in enumerate(x, 1):
            # print("my Vdotx shape is ")
            # print(self.V.dot(x).shape)

            # print("my udoths is ")
            # print(self.U.dot(self.hs[i-1]).shape)

            # print("my sum shape is")
            # print((self.V.dot(x.reshape(1,-1)) + self.U.dot(self.hs[i-1])).shape)

            h_t = np.tanh(self.V.dot(x.reshape(1,-1)) + self.U.dot(self.hs[i-1]))
            print(f"my h_t shape is {h_t.shape}")
            yhat_t = self.w.dot(h_t)
            self.hs[i] = h_t
            print(type(yhat_t))
            print(yhat_t.shape)
            print(yhat_t)
            self.yhats[i] = float(yhat_t) 

        # return last saved yhat & h time step 
        return h_t, yhat_t 

    def SGD(self,X, y, batch_size, epochs, epsilon):
        # randomize training set
        # self.SGD_param = [batch_size, epochs, epsilon, alpha]
        permute = np.random.permutation(X.shape[1])
        shuffled_X  = X.T[permute].T #(1, 50)
        # shuffled_y = self.y.T[permute].T #(10,55000)
        shuffled_y = y.T[permute].T

        sample_size = X.shape[0] # total batch size

        # get all indexes based on batch size
        rounds = get_indexes(sample_size, batch_size) # list of (start,end) for slicing each batch X

        print(rounds)

        # start iteration loop
        for epoch in range(1, epochs+1):
            # print(f"Epoch [{epoch}]")
            for indexes in rounds:
                start, finish = indexes
                self.forward(shuffled_X[:,start:finish])
                gradient_V, gradient_U, gradient_w = self.backward(shuffled_y[:,start:finish])
                self.V -= epsilon * gradient_V
                self.U -= epsilon * gradient_U
                self.w -= epsilon * gradient_w

            yhat = self.foward(X)
            MSE_val = MSE(yhat, y)
            print("This is epoch round [{}]".format(epoch))
            print("Epoch [{}], MSE Loss: {}".format(epoch , MSE_valE))

def MSE (yhat, y):
    coeff = 1 / (2 * X_tilde.shape[1])
    sum = np.sum((yhat - y)**2)
    mse = coeff * sum
    return mse

# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    xs, ys = generateData()
    # print(xs)
    # print(ys)
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    print(numTimesteps)
    rnn = RNN(numHidden, numInput, 1)

    batch_size = 1
    epsilon = 0.01
    epochs = 10

    rnn.SGD(np.array(xs).reshape(1,-1), np.array(ys).reshape(1,-1), batch_size, epochs,epsilon)

    # rnn2 = rnn.forward(np.array(xs).reshape(1,-1))
    # print(rnn.backward(ys))

    # print(rnn.yhats[50])
    # TODO: IMPLEMENT ME

