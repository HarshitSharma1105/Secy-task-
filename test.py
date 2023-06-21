
import numpy as np
import random
import pickle
import os

# (train_X, train_y), (test_X, test_y) = datasets.cifar10.load_data()
# print(train_X.shape)
# print(train_y.shape)
# print(test_X.shape)
# print(test_y.shape)
# train_X.reshape(50000,3,32,32)
# test_X.reshape(10000,3,32,32)
# train_X = train_X/255
# test_X = test_X/255
# train_data=list(zip(train_X,train_y))
# test_data=list(zip(test_X,test_y))

def unpickle(file):
    with open(file,'rb') as fo:
        dict=pickle.load(fo,encoding='bytes')
    return dict


file1=r"C:\Users\Harshit Sharma\ML Task\data_batch_1"
data_batch_1 = unpickle(file1)

file2=r"C:\Users\Harshit Sharma\ML Task\data_batch_2"
data_batch_2 = unpickle(file2)


file3=r"C:\Users\Harshit Sharma\ML Task\data_batch_3"
data_batch_3 = unpickle(file3)


file4=r"C:\Users\Harshit Sharma\ML Task\data_batch_4"
data_batch_4 = unpickle(file4)


file5=r"C:\Users\Harshit Sharma\ML Task\data_batch_5"
data_batch_5= unpickle(file5)

testfile=r"C:\Users\Harshit Sharma\ML Task\data_batch_5"
data_test=unpickle(testfile)

data_batch_1[b'data'] = np.reshape(data_batch_1[b'data'],(10000,32,32,3))
data_batch_2[b'data'] = np.reshape(data_batch_2[b'data'],(10000,32,32,3))
data_batch_3[b'data'] = np.reshape(data_batch_3[b'data'],(10000,32,32,3))
data_batch_4[b'data'] = np.reshape(data_batch_4[b'data'],(10000,32,32,3))
data_batch_5[b'data'] = np.reshape(data_batch_5[b'data'],(10000,32,32,3))
x1 = data_batch_1[b'data']
x2 = data_batch_2[b'data']
x3 = data_batch_3[b'data']
x4 = data_batch_4[b'data']
x5 = data_batch_5[b'data']
y1 = data_batch_1[b'labels']
y2 = data_batch_2[b'labels']
y3 = data_batch_3[b'labels']
y4 = data_batch_4[b'labels']
y5 = data_batch_5[b'labels']
yt = data_test[b'labels']
y = y1+y2+y3+y4+y5
con = np.concatenate((x1,x2,x3,x4,x5),axis=0)
con = con.reshape(50000,3,32,32)
test = data_test[b'data']
test = test.reshape(10000,3,32,32)





def act(z,activation):
  if(activation=='sigmoid'):
    return 1/(1+np.exp(-z))
  elif(activation=='tanh'):
    return np.tanh(z)

def act_der(z,activation):
  if(activation=='sigmoid'):
    return z(1-z)
  elif(activation=='tanh'):
    return 1-np.tanh(z)**2



class Network(object):
    def __init__(self,sizes,anneal,optimizer,activation,momentum,mini_batch_size,loss,learning_rate): # sizes is a list containing the network.
                              # eg : [784,128,10] means input =784 neurons,
        weights_velocities=[]
        biases_velocities=[]                     #    1st hidden layer 128 neurons, output 10 neurons.
        sizes.insert(0,1024)
        sizes.append(10)
        self.sizes=sizes
        self.num_layers=len(sizes)
        w = np.ones((3,32,32))
        b = np.zeros((32,32))
        weights = []
        bias = []
        weights.append(w)
        bias.append(b)
        weights.append(np.random.randn(sizes[0],32**2))
        bias.append(np.random.randn(sizes[0],1))
        for i in range(len(sizes)-1):
            weights.append(np.random.randn(sizes[i+1],sizes[i]))
            bias.append(np.random.randn(sizes[i+1],1))
        weights.append(np.random.randn(10,sizes[-1]))
        bias.append(np.random.randn(10,1))
        self.activation=activation
        self.anneal=anneal
        self.optimizer=optimizer
        self.momentum=momentum
        self.batchsize=mini_batch_size
        self.learning_rate=learning_rate
        self.loss=loss
        self.weights_velocities=weights_velocities
        self.biases_velocities=biases_velocities
        self.beta1=0.9
        self.beta2=.99
        self.epsilon=1e-8

    def flatten(self,x):
        x = (x-np.min(x))/(np.max(x)-np.min(x))
        x = x*2 - 1
        sum = np.zeros((x.shape[1],x.shape[1]))
        for i in range(3):
            sum += np.matmul(x[i],self.weights[0][i].T)
        sum += self.bias[0]
        a = sum.reshape((sum.shape[0]**2,1))
        self.a = a
        return a


    def show(self):
      print(self.num_layers)
      for bias in self.biases:
        print(bias.shape)
      for weight in self.weights:
         print(weight.shape)


    def forwardpropagation(self,a):
        a=self.flatten(a)
        for b,w in zip(self.biases, self.weights):
            a=act(np.matmul(w,a)+b,self.activation)
        return a




    def backpropagation(self,x,y):

        y_t = np.zeros((len(y), 10))
        y_t[np.arange(len(y)), y] = 1
        y_t= y_t.T


        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]


        activation=x
        activation_list=[x]


        for w,b in zip(self.weights,self.biases):
            activation= act(np.matmul(w,activation)+b,self.activation)
            activation_list.append(activation)

        delta= activation_list[-1]-y_t
        m=self.sizes[-1]


        nabla_w[-1]=np.matmul(delta,activation_list[-2].T)
        nabla_b[-1]=(np.sum(delta,axis=1).reshape(m,1))

        for j in range(2,self.num_layers):
            delta= np.matmul(self.weights[-j+1].T,delta)*act_der(activation[-j],self.activation)

            nabla_b[-j]= (np.sum(delta,axis=1).reshape(self.sizes[-j],1))*(1/m)
            nabla_w[-j]= np.matmul(delta,activation_list[-j-1].T)*(1/m)
        return (nabla_b,nabla_w)

    def SGD(self, train_data,epochs,mini_batch_size):
        n_train= len(train_data)
        for i in range(epochs):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k+ mini_batch_size] for k in range(0,n_train,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)

            self.predict(train_data)
            print("Epoch {0} completed.".format(i+1))

    def predict(self,test_data):
        test_results = [(np.argmax(self.forwardpropagation(x)),y) for x,y in test_data]
        # returns the index of that output neuron which has highest activation

        num= sum(int (x==y) for x,y in test_results)
        print ("{0}/{1} classified correctly.".format(num,len(test_data)))

    def gradient_descent(self,minibatch):
      weights=self.weights
      biases=self.biases
      for x,y in minibatch:
          db,dw=self.backpropagation(x,y)
          weights=[w+dw for w,dw in zip(weights,dw)]
          biases=[b+db for b,db in zip(biases,db)]


    def momentum_update(self,minibatch):
     vel_b=[np.zeros(b.shape) for b in self.bias]
     vel_w=[np.zeros(w.shape) for w in self.weights]
     grad_w=[np.zeros(w.shape) for w in self.weights]
     grad_b=[np.zeros(b.shape) for b in self.bias]

     j=0
     for x,y in minibatch:
            j += 1
            delta_b_i,delta_w_i = self.backpropagation(x,y,self)
            delta_b = [nb+db for nb,db in zip(delta_b,delta_b_i)]
            delta_w = [nw+dw for nw,dw in zip(delta_w,delta_w_i)]
            if j%self.batchsize==0:
                vel_b=[self.beta1*nb + (self.learning_rate*db)/j for nb,db in zip(vel_b,grad_b)]
                vel_w=[self.beta1*nw + (self.learning_rate*dw)/j for nw,dw in zip(vel_w,grad_w)]
                self.weights=[w- nw for w,nw in zip(self.weights,vel_w)]  #averaging over many example
                self.bias=[b-nb for b,nb in zip(self.bias,vel_b)]
           
    def nag_update(self,minibatch):
        grad_b = [np.zeros(b.shape) for b in self.bias]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        total_iterations = 0

        for x, y in minibatch:
            grad_b, grad_w = self.backpropagation(x, y,self.fun)
            total_iterations += 1

            nabla_b = [self.beta1 * nb + (self.learning_rate * db) / total_iterations for nb, db in zip(nabla_b, grad_b)]
            nabla_w = [self.beta1* nw + (self.learning_rate * dw) / total_iterations for nw, dw in zip(nabla_w, grad_w)]

            if total_iterations % self.mini_batch_size == 0:
                self.weights = [w - self.beta * vw for w, vw in zip(self.weights, nabla_w)]
                self.bias = [b - self.beta * vb for b, vb in zip(self.bias, nabla_b)]

                grad_b, grad_w = self.backpropagation(x, y, self.fun)
                nabla_b = [self.beta * nb + (self.lr * db) / total_iterations for nb, db in zip(nabla_b, grad_b)]
                nabla_w = [self.beta * nw + (self.lr * dw) / total_iterations for nw, dw in zip(nabla_w, grad_w)]

                self.weights = [w - nw for w, nw in zip(self.weights, nabla_w)]  # averaging over many examples
                self.bias = [b - nb for b, nb in zip(self.bias, nabla_b)]

    def adam_update(self,minibatch):
        i = 0
        delta_b = [np.zeros(b.shape) for b in self.bias]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        res_w = [np.zeros(w.shape) for w in self.weights]
        res_b = [np.zeros(b.shape) for b in self.bias]
        for x,y in minibatch:
            grad_b,grad_w = self.backpropagation(x,y,self.fun)
            i+=1
            res_w = [self.beta1*nw + (1-self.beta1)*dw for nw,dw in zip(nabla_w,grad_w)]
            res_b = [self.beta1*nb + (1-self.beta1)*db for nb,db in zip(nabla_b,grad_b)]
            nabla_w = [self.beta2*nw + (1-self.beta2)*dw**2 for nw,dw in zip(nabla_w,grad_w)]
            nabla_b = [self.beta2*nb + (1-self.beta2)*db**2 for nb,db in zip(nabla_b,grad_b)]
            if i%self.mini_batch_size==0:
                res_w = res_w/(1-self.beta1**i)
                res_b = res_b/(1-self.beta1**i)
                nabla_w = nabla_w/(1-self.beta2**i)
                nabla_b = nabla_b/(1-self.beta2**i)
                for i in range(len(self.weights)):
                    self.weights[i] -= (self.learning_rate/np.sqrt(nabla_w[i]+self.epsilon))*res_w[i]
                    self.bias[i] -= (self.learning_rate/np.sqrt(nabla_b[i]+self.epsilon))*res_b[i]  


    def update_mini_batch(self,mini_batch):
        n_train= len(self.train_data)
        for i in range(100):
            np.random.shuffle(self.train_data)
            mini_batches = [self.train_data[k:k+ self.mini_batch_size] for k in range(0,n_train,self.batchsize)]
            if self.optimizer == "gd":
                for mini_batch in mini_batches:
                    self.gradient_descent(mini_batch)
            elif self.optimizer == "momentum":
                for mini_batch in mini_batches:
                    self.momentum_update(mini_batch)
            elif self.optimizer == "nag":
                for mini_batch in mini_batches:
                    self.nag_update(mini_batch)
            elif self.optimizer=="adam":
                for mini_batch in mini_batches:
                    self.adam_update(mini_batch,)
            num = 0
            with open("log_train.txt",'a') as f:
                f.write(f"Epoch {i+1}, Step {j}, Loss: {self.training_batch_loss(self.loss,self.train_data[j-100:j])}, Error:{(1-num/j)*100}, lr: {self.lr}\n")

            for j in range(1,len(self.train_data)+1):
                a = self.predict(self.train_data[j-1])
                if a==1:
                    num += 1
                if j%100 == 0:
                    with open("log_train.txt","a") as f:
                        f.write(f"Epoch {i+1}, Step {j}, Loss: {self.training_batch_loss(self.loss,self.train_data[j-100:j])}, Error:{(1-num/j)*100}, lr: {self.lr}\n")

            self.anneal(self.lr)
