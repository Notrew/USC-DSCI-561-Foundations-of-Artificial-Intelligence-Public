import math
import numpy as np
import sys

class MLP:
    def __init__(self,layers,mini_batch_size=32,learning_rate=0.1,num_epoch=100):
        self.layers = layers
        self.num_layer = len(self.layers)
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch

        self.params = {}
        self.instance_num = 0
    
    # intialize weights and biases parameters
    def initialize_parameters(self):
        np.random.seed(1)
        for l in range(1, len(self.layers)):
            self.params[f"W{l}"] = np.random.randn(self.layers[l],self.layers[l-1])
            self.params[f"b{l}"] = np.zeros((self.layers[l],1))

    # define activation function
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    
    def softmax(self,x):
        exps = np.exp(x - x.max())
        return exps/np.sum(exps,axis=0)

    def sigmoid_derivative(self,Z):
        sig = self.sigmoid(Z)
        return sig*(1-sig)

    # define forward propagation
    def forward_propagation(self,X):
        cache = {}
        A = X
        # iteratelly computation
        # cache is like:{A1,W1,Z1,...,}
        for l in range(self.num_layer):
            Z = self.params[f"W{l+1}"].dot(A)+self.params[f"b{l+1}"]
            A = self.sigmoid(Z)
            # stored in dictionary
            cache[f"A{l+1}"] = A
            cache[f"W{l+1}"] = self.params["W"+str(l+1)]
            cache[f"Z{l+1}"] = Z
        
        return A,cache

    # define backward propagation
    def backward_propagation(self,X,Y,cache):
        # last_layer = self.num_layer
        self.instance_num = X.shape[1]
        derivatives = {}
        cache["A0"] = X

        # at last layer
        # res of last layer
        A = cache[f"A{self.num_layer}"]
        # the derivative of the loss with respect to the output res
        dA = -np.divide(Y,A) + np.divide(1 - Y,1 - A)
        dZ_last_layer = dA*self.sigmoid_derivative(cache[f"Z{self.num_layer}"])
        dW_last_layer = dZ_last_layer.dot(cache[f"A{self.num_layer-1}"].T)/self.instance_num
        db_last_layer = np.sum(dZ_last_layer,axis=1,keepdims=True)/self.instance_num
        # stored in dictionary
        derivatives[f"dW{self.num_layer}"] = dW_last_layer
        derivatives[f"db{self.num_layer}"] = db_last_layer

        # iteratelly compute on remaining layres
        dAPrev = cache[f"W{self.num_layer}"].T.dot(dZ_last_layer)
        for l in range(self.num_layer-1,0,-1):
            dZ = dAPrev*self.sigmoid_derivative(cache[f"Z{l}"])
            dW = dZ.dot(cache[f"A{l-1}"].T)/self.instance_num
            db = np.sum(dZ,axis=1,keepdims=True)/self.instance_num
            derivatives[f"dW{l}"] = dW
            derivatives[f"db{l}"] = db
            if l > 1:
                dAPrev = cache[f"W{l}"].T.dot(dZ)

        return derivatives

    def get_mini_batches(self,X,Y,mini_batch_size):
        m = X.shape[1]
        mini_batches = []
        num_batches = math.floor(m/mini_batch_size)
        for i in range(num_batches):
            mb_X = X[:,i*mini_batch_size:(i+1)*mini_batch_size]
            mb_Y = Y[:,i*mini_batch_size:(i+1)*mini_batch_size]
            mini_batch = (mb_X,mb_Y)
            mini_batches.append(mini_batch)
        # if aliquant
        if m % mini_batch_size != 0:
            mb_X = X[:,mini_batch_size*num_batches:m]
            mb_Y = Y[:,mini_batch_size*num_batches:m]
            mini_batch = (mb_X,mb_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def fit(self,X,Y):
        np.random.seed(1)
        learning_rate = self.learning_rate
        self.instance_num = X.shape[1]
        self.layers.insert(0,X.shape[0])

        self.initialize_parameters()
        for epoch in range(self.num_epoch):
            # shuffle X and y
            random = np.arange(self.instance_num)
            np.random.shuffle(random)
            X_shuffle = X[:,random]
            Y_shuffle = Y[:,random]

            # get mini-batches
            mini_batches = self.get_mini_batches(X_shuffle,Y_shuffle,self.mini_batch_size)
            # mini_batches = self.get_mini_batches(X,Y,self.mini_batch_size)
            for mini_batch in mini_batches:
                mb_x, mb_y = mini_batch
                A,cache = self.forward_propagation(mb_x)
                derivatives = self.backward_propagation(mb_x,mb_y,cache)
                # update parameters
                for l in range(1,self.num_layer + 1):
                    self.params[f"W{l}"] = self.params[f"W{l}"]-learning_rate*derivatives[f"dW{l}"]
                    self.params[f"b{l}"] = self.params[f"b{l}"]-learning_rate*derivatives[f"db{l}"]

    def predict(self,X):
        A,_= self.forward_propagation(X)
        m = X.shape[1]
        pred = np.zeros((1,m))
        for i in range(A.shape[1]):
            if A[0,i] > 0.5:
                pred[0,i] = 1
            else:
                pred[0,i] = 0
        return pred
    
    # def accuracy_score(self,X,Y):
    #     A,_ = self.forward_propagation(X)
    #     m = X.shape[1]
    #     pred = np.zeros((1,m))
    #     for i in range(A.shape[1]):
    #         if A[0,i] > 0.5:
    #             pred[0,i] = 1
    #         else:
    #             pred[0,i] = 0
    #     return np.sum((pred == Y)/m)

# read data and fit dimensions of model
x_train = np.loadtxt(sys.argv[1],delimiter = ",").T
y_train = np.loadtxt(sys.argv[2],delimiter = ",",ndmin=2).T.astype("int")
x_test = np.loadtxt(sys.argv[3],delimiter = ",").T
# y_test = np.loadtxt(sys.argv[4],delimiter = ",",ndmin=2).T.astype("int")

# add sin(x) as new features
x_train = np.vstack((x_train,np.sin(x_train)))
x_test = np.vstack((x_test,np.sin(x_test)))
layers_dims = [4,100,50,1]

# layers_dims = [2,100,50,1]
neural_network = MLP(layers_dims,mini_batch_size=32,learning_rate=0.3,num_epoch=1000)
neural_network.fit(x_train,y_train)
y_pred = neural_network.predict(x_test).T.astype("int")

# print("Testing Accuracy is :",neural_network.accuracy_score(x_test,y_test))
# save the predictions if test set
np.savetxt("test_predictions.csv",y_pred)

# python3 NeuralNetwork3.py circle_train_data.csv circle_train_label.csv circle_test_data.csv circle_test_label.csv
# python3 NeuralNetwork3.py spiral_train_data.csv spiral_train_label.csv spiral_test_data.csv spiral_test_label.csv