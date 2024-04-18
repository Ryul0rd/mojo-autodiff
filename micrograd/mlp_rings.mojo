from engine import Value, SGD, argmax
from nn import MLP, cross_entropy
from progress import ProgressBar

from math import sin, cos
from random import random_float64, randn_float64, random_si64, seed


alias PI = 3.1415926


fn main():
    alias LEARNING_RATE = 1e-4
    alias N_EPOCHS = 40

    alias radii = List[Float64](0, 1, 2)
    alias variances = List[Float64](0.25, 0.25, 0.25)

    var train_data = List[RingSample]()
    var test_data = List[RingSample]()
    try:
        train_data = ring_dataset(1000, radii, variances)
        test_data = ring_dataset(1000, radii, variances)
    except:
        print("Couldn't create data")
        return

    # model
    var model = MLP(in_features=2, hidden_sizes=List(32), out_features=len(radii))
    var optimizer = SGD(model.parameters(), learning_rate=LEARNING_RATE)

    # training loop
    for epoch in range(N_EPOCHS):
        # train
        var training_progress = ProgressBar(len(train_data), desc='Training')
        for i_train in range(len(train_data)):
            var x = List(Value(train_data[i_train].x1), Value(train_data[i_train].x2))
            var y_true = train_data[i_train].y
            var logits = model(x)
            var loss = cross_entropy(logits, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_progress.update()
        # validate
        var total_test_loss: Float32 = 0.
        var n_correct = 0
        var validation_progress = ProgressBar(len(test_data), desc='Validating')
        for i_test in range(len(test_data)):
            var x = List(Value(test_data[i_test].x1), Value(test_data[i_test].x2))
            var y_true = test_data[i_test].y
            var logits = model(x)
            var loss = cross_entropy(logits, y_true)
            total_test_loss += loss.data[]
            # some bug causes lifetime of loss to end early if not for next line
            _ = loss
            var y_pred = argmax(logits)
            n_correct += 1 if y_pred == y_true else 0
            validation_progress.update()
        print('Epoch ' + String(epoch) + ' Test Loss: ' + String(total_test_loss / len(test_data)))
        print('Epoch ' + String(epoch) + ' Test Acc : ' + String(n_correct / len(test_data)))


fn ring_dataset(size: Int, radii: List[Float64], variances: List[Float64]) raises -> List[RingSample]:
    if len(radii) != len(variances):
        raise Error('radii and variances must be same size')
    var data = List[RingSample](capacity=size)
    for _ in range(size):
        var label = int(random_si64(0, len(radii)-1))
        var radius = randn_float64(radii[label], variances[label])
        var azimuth = random_float64(0, 2*PI)
        var x1 = radius * cos(azimuth)
        var x2 = radius * sin(azimuth)
        data.append(RingSample(x1, x2, label))
    return data


@value
struct RingSample:
    var x1: Float32
    var x2: Float32
    var y: Int 
