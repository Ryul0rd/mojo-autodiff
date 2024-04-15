from engine import Value, SGD
from nn import MLP, cross_entropy
from progress import ProgressBar

from time import now


fn main():
    # hparams
    alias LEARNING_RATE = 3e-4
    alias N_EPOCHS = 2

    # data
    var x_train = List[List[UInt8]]()
    var x_test = List[List[UInt8]]()
    var y_train = List[Int8]()
    var y_test = List[Int8]()
    try:
        print('Loading training images...')
        x_train = read_images('data/mnist_train.csv')
        print('Loading test images...')
        x_test = read_images('data/mnist_test.csv')
        print('Loading training labels...')
        y_train = read_labels('data/mnist_train.csv')
        print('Loading test labels...')
        y_test = read_labels('data/mnist_test.csv')
    except:
        print('Error reading data. Aborting run.')
        return

    # model
    print('Initializing model')
    var model = MLP(in_features=28*28, hidden_sizes=List(32, 32), out_features=10)
    var optimizer = SGD(model.parameters(), learning_rate=LEARNING_RATE)

    # train
    print('Beginning training')
    var iteration = 0
    for _ in range(N_EPOCHS):
        for i_train in range(len(x_train)):
            if (i_train + 1) % 30000 == 0 or i_train == 0:
                var total_test_loss: Float32 = 0.
                var n_correct = 0
                var validation_progress = ProgressBar(len(x_test), 'Validating')
                for i_test in range(len(x_test)):
                    var image = image_to_values(x_test[i_test])
                    var label = int(y_test[i_test])
                    var logits = model(image)
                    var loss = cross_entropy(logits, label)
                    total_test_loss += loss.data[]
                    var pred = argmax(logits)
                    n_correct += 1 if pred == label else 0
                    validation_progress.update()
                print('Iteration ' + String(i_train) + ' Test Loss: ' + String(total_test_loss / len(x_test)))
                print('Iteration ' + String(i_train) + ' Test Acc : ' + String(n_correct / len(x_test)))
            break


fn image_to_values(image: List[UInt8]) -> List[Value]:
    var out = List[Value](capacity=len(image))
    for pixel in image:
        out.append(Value(int(pixel[])/255))
    return out


fn read_images(path: String) raises -> List[List[UInt8]]:
    var content = String('')
    with open(path, mode='r') as f:
        content = f.read()
    var rows = content[:-1].split('\n')[1:]
    var images = List[List[UInt8]](capacity=10000)
    for row in rows:
        var pixel_strings = row[][2:].split(',')
        var image = List[UInt8](capacity=28*28)
        for pixel_string in pixel_strings:
            image.append(atol(pixel_string[].strip()))
        images.append(image)
    return images


fn read_labels(path: String) raises -> List[Int8]:
    var content = String('')
    with open(path, mode='r') as f:
        content = f.read()
    var lines = content.split('\n')[1:]
    var labels = List[Int8]()
    for line in lines:
        if line[] == '':
            continue
        var label = atol(line[].split(',')[0])
        labels.append(label)
    return labels


fn argmax(vec: List[Value]) -> Int8:
    var highest_index = -1
    var highest_value = Float32.MIN_FINITE
    for i in range(len(vec)):
        if vec[i] > highest_value:
            highest_value = vec[i].data[]
            highest_index = i
    return highest_index
