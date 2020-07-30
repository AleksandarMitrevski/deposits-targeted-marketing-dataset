import random

full_ds = open('../../dataset/bank-additional-full.csv', 'r') 
lines = full_ds.readlines()
full_ds.close()
header = [lines[0]]
data_lines = lines[1:]

test_percentage = 1 / 10
training_reduced_percentage = 15 / 100

test_partition_count = round(test_percentage * len(data_lines))
training_partition_count = len(data_lines) - test_partition_count

random.shuffle(data_lines)
training_lines = data_lines[0:training_partition_count+1]
test_lines = data_lines[training_partition_count+1:len(data_lines)]

training_ds = open('../data/training.csv', 'w+')
training_ds.writelines(header + training_lines)
training_ds.close()

test_ds = open('../data/test.csv', 'w+')
test_ds.writelines(header + test_lines)
test_ds.close()

random.shuffle(training_lines)
training_reduced_count = round(training_reduced_percentage * len(training_lines))
training_reduced_out_lines = training_lines[0:training_reduced_count+1]

training_reduced_ds = open('../data/training_reduced.csv', 'w+')
training_reduced_ds.writelines(header + training_reduced_out_lines)
training_reduced_ds.close()
