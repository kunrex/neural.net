import csv

from src.network import NeuralNetwork

batch_count = 30
test_count = 28 * 28

def read_data(reader):
    next(reader)
    batches = []
    current = []

    i = 1
    for line in reader:
        test = [[1 if (int(line[0]) == (x + 1)) else 0 for x in range(0, 10)]]
        test.extend([int(x) / 255 for x in line[1::]])

        current.append(test)
        if i % batch_count == 0:
            batches.append(list(current))
            current.clear()
            i = 1

        i += 1

    return batches

dataset = open('./../../datasets/digits/mnist_train.csv')
tests = open('./../../datasets/digits/mnist_test.csv')

network = NeuralNetwork(2, 16, test_count, 10).initialise()

train_batches =  read_data(csv.reader(dataset))
dataset.close()

test_batches = read_data(csv.reader(tests))
tests.close()

for batch in train_batches:
  network.train(batch)

tests = []
for batch in test_batches:
    for test in batch:
        tests.append(test)

network.test(tests)



