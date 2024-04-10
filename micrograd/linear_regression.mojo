from engine import Value, SGD, pointer_init, mse
from random import random_float64


fn main():
    # hparams
    alias TRAIN_SIZE = 1000
    alias TEST_SIZE = 100
    alias TRUE_WEIGHT = 2.5
    alias TRUE_BIAS = 0.7

    alias LEARNING_RATE = 2e-2
    alias N_EPOCHS = 1

    # data
    var train_set = List[Sample]()
    var test_set = List[Sample]()
    for _ in range(TRAIN_SIZE):
        var x = random_float64(-1, 1)
        var y = x * TRUE_WEIGHT + TRUE_BIAS
        train_set.append(Sample(x, y))
    for _ in range(TEST_SIZE):
        var x = random_float64(-1, 1)
        var y = x * TRUE_WEIGHT + TRUE_BIAS
        test_set.append(Sample(x, y))

    # model
    var w = Value(1)
    var b = Value(0)
    var params = List[Value]()
    params.append(w)
    params.append(b)

    # training loop
    var optimizer = SGD(params, learning_rate=LEARNING_RATE)
    var iteration = 0
    for _ in range(N_EPOCHS):
        for train_sample in train_set:
            # test every x iterations
            if iteration % 100 == 0:
                var test_loss_acc: Float32 = 0
                for test_sample in test_set:
                    var x = Value(test_sample[].x)
                    var y_true = Value(test_sample[].y)
                    var y_pred = w * x + b
                    var loss = mse(y_pred, y_true)
                    test_loss_acc += loss.data[]
                print(String('Iteration ') + iteration + ' test loss: ' + (test_loss_acc / TEST_SIZE))
            # train
            var x = Value(train_sample[].x)
            var y_true = Value(train_sample[].y)
            var y_pred = w * x + b
            var loss = mse(y_pred, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iteration += 1
    
    print(String('True Weight   : ') + TRUE_WEIGHT)
    print(String('Learned Weight: ') + w.data[])
    print(String('True Bias     : ') + TRUE_BIAS)
    print(String('Learned Bias  : ') + b.data[])

@value
struct Sample:
    var x: Float32
    var y: Float32
