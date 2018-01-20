import numpy as np

class neural_net:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.eta = 1E-3
        self.momentum = 2E-4

        self.weights1 = 2*np.random.rand(40, nhidden)-1
        self.weights2 = 2*np.random.rand(nhidden, 8)-1

        print("INITIALIZING NETWORK WITH %d HIDDEN LAYER NEURONS." % (nhidden))

    # sigmoid activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # calculate sigmoid derivative
    def sigmoidGrad(self,x):
        return np.exp(-x)/((1+np.exp(-x))**2)

    # calculate cost
    def cost(self, actual, targets):
        return 0.5*np.sum((targets-actual)**2)

    # use validation data set to stop network from overfitting (calls train)
    def earlystopping(self, inputs, targets, valid, validtargets):
        valid_costs = [float("inf")]*5
        avg = float("inf") # current moving average of costs
        prev_avg = avg # previous moving average of costs
        epoch = 0

        print('BEGIN TRAINING.\n')
        # train as long as moving average does not increase
        while avg <= prev_avg:
            prev_avg = avg
            train_cost = self.train(inputs, targets)

            # forward prop with validation data
            out_valid = self.forward(valid)[1]

            # update moving average
            v_cost = self.cost(out_valid, validtargets)
            valid_costs.append(v_cost)
            if len(valid_costs) > 5:
                valid_costs.pop(0)
            avg = np.sum(valid_costs)/5

            # print information
            epoch +=1
            print('Epoch %d (%d iterations):' % (epoch, epoch * 10))
            print('Training Cost: %.3f' % (train_cost))
            print('Validation Cost: %.3f\n' % (v_cost))
        print("TRAINING COMPLETE. BEGIN TESTING.\n")

    # train the network
    def train(self, inputs, targets, iterations=10):
        cost = 0
        for iters in range(iterations):
            hid, out = self.forward(inputs)
            cost = self.cost(out, targets)

            # error and gradient of the output layer
            err_out = out - targets
            grad_out = np.dot(hid.T, err_out)

            #error and gradient of the input layer
            err_hid = np.dot(err_out, self.weights2.T)*self.sigmoidGrad(hid)
            grad_hid = np.dot(inputs.T, err_hid)

            #update weights (uses momentum)
            self.weights2 -= (self.eta*grad_out + self.weights2*self.momentum)
            self.weights1 -= (self.eta*grad_hid + self.weights1*self.momentum)
        return cost

    # run the network
    def forward(self, inputs):
        hid_layer = np.dot(inputs, self.weights1)
        hid_act = self.sigmoid(hid_layer)
        out_layer = np.dot(hid_act, self.weights2)
        return hid_act, out_layer


    # run the network with testing data set, print confusion matrix
    def confusion(self, inputs, targets):
        print("Printing Confusion Matrix (%d entries):" % (len(inputs)))
        conf_matrix = np.zeros((8,8))
        predict = self.forward(inputs)[1]
        num_correct = 0
        for i in range(len(inputs)):
            guess = predict[i,:]
            truth = targets[i,:]
            guess_class = [index for index, value in enumerate(guess) if value == max(guess)]
            true_class = [index for index, value in enumerate(truth) if value == max(truth)]
            if guess_class == true_class:
                num_correct += 1
            np.set_printoptions(precision=1)
            np.set_printoptions(suppress=True)
            conf_matrix[guess_class, true_class] += 1

        # format confusion matrix to display percentages of all data
        conf_percent = conf_matrix * 1/(0.01*len(inputs))
        conf_percent = conf_percent.tolist()
        conf_percent = [["{:4.1f}%".format(conf_percent[i][j]) for i in range(8)] for j in range(8)]
        for arr in conf_percent:
            print(arr)
        print("Correct classifications: %d/%d (%.1f%%)" % (num_correct, len(inputs), num_correct/(0.01*len(inputs))))
        return num_correct, conf_matrix
