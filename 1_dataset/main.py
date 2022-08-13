import numpy as np
from PIL import Image
import glob
import os, os.path
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import json
np.random.seed(0)

def main():
    with open('config.json') as f:
        params = json.load(f)
    save_dataset_again = False
    dataset_name = params["dataset_name"]  # MNIST, CIFAR10
    n_samples_train, n_samples_test = 2000, 2000
    samples_of_class_per_batch = params["samples_of_class_per_batch"]
    n_classes = params["n_classes"]
    path_to_save_np = f'.\\dataset\\{dataset_name}\\np\\'
    path_to_save_png = f'.\\dataset\\{dataset_name}\\png\\'
    path_save_pickle = f'.\\dataset\\{dataset_name}\\batches_{n_classes}_classes\\'

    batch_size = n_classes * samples_of_class_per_batch
    if dataset_name == "MNIST":
        X_train, X_test, y_train, y_test = read_MNIST_dataset(n_samples_train, n_samples_test)
    elif dataset_name == "CIFAR10":
        X_train, X_test, y_train, y_test = read_CIFAR10_dataset(n_samples_train, n_samples_test)

    X_train_classes, X_test_classes, y_train_classes, y_test_classes = separate_classes(X_train, X_test, y_train, y_test)
    assert n_classes <= len(X_train_classes), 'The selected number of classes is more than the number of existing classes in dataset!'
    if save_dataset_again:
        prepare_dataset(X_train_classes, X_test_classes, path_to_save_np, path_to_save_png)
    prepare_batches(batch_size, n_classes=n_classes, path_np_data=path_to_save_np, path_save_pickle=path_save_pickle, train_data=True)
    prepare_batches(batch_size, n_classes=n_classes, path_np_data=path_to_save_np, path_save_pickle=path_save_pickle, train_data=False)

def prepare_batches(batch_size, n_classes, path_np_data, path_save_pickle, train_data=True):
    n_samples_of_class_per_batch = batch_size // n_classes
    i = 0
    batches = []
    while True:
        batch = []
        finished = False
        for class_index in range(n_classes):
            if train_data:
                paths_of_samples_of_class = glob.glob(path_np_data+f'train\\{class_index}\\'+'*.npy')
            else:
                paths_of_samples_of_class = glob.glob(path_np_data+f'test\\{class_index}\\'+'*.npy')
            if (i+1)*n_samples_of_class_per_batch >= len(paths_of_samples_of_class):
                finished = True
                break
            paths_of_samples_of_class_in_batch = paths_of_samples_of_class[i*n_samples_of_class_per_batch:(i+1)*n_samples_of_class_per_batch]
            batch.extend(paths_of_samples_of_class_in_batch)
        if finished:
            break
        batches.append(batch)
        i += 1
    if not os.path.exists(path_save_pickle):
        os.makedirs(path_save_pickle)
    if train_data:
        with open(path_save_pickle+'batches_train.pickle', 'wb') as handle:
            pickle.dump(batches, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path_save_pickle+'batches_test.pickle', 'wb') as handle:
            pickle.dump(batches, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return batches

def prepare_dataset(X_train_classes, X_test_classes, path_to_save_np, path_to_save_png):
    n_classes = len(X_train_classes)
    # prepare training data:
    for class_index in range(n_classes):
        n_samples_of_class = X_train_classes[class_index].shape[0]
        shuffled_indices = np.random.permutation(n_samples_of_class)
        for index in shuffled_indices:
            X = X_train_classes[class_index][index, :, :]
            save_numpy_array(path_to_save=path_to_save_np+f'train/{class_index}/', arr_name=f'{index}', arr=X)
            save_array_as_image(path_to_save=path_to_save_png+f'train/{class_index}/', arr_name=f'{index}', arr=X)
                
    # prepare val/test data:
    for class_index in range(n_classes):
        n_samples_of_class = X_test_classes[class_index].shape[0]
        shuffled_indices = np.random.permutation(n_samples_of_class)
        for index in shuffled_indices:
            X = X_test_classes[class_index][index, :, :]
            save_numpy_array(path_to_save=path_to_save_np+f'test/{class_index}/', arr_name=f'{index}', arr=X)
            save_array_as_image(path_to_save=path_to_save_png+f'test/{class_index}/', arr_name=f'{index}', arr=X)

def save_numpy_array(path_to_save, arr_name, arr):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    np.save(path_to_save+arr_name+".npy", arr)

def save_array_as_image(path_to_save, arr_name, arr):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    im = Image.fromarray(arr)
    im.save(path_to_save+arr_name+".png")

def read_MNIST_dataset(n_samples_train=None, n_samples_test=None):
    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data?version=stable
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    #######################
    if n_samples_train is not None:
        X_train = X_train[:n_samples_train, :, :]
        y_train = y_train[:n_samples_train]
    if n_samples_test is not None:
        X_test = X_test[:n_samples_test, :, :]
        y_test = y_test[:n_samples_test]
    #######################
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # plt.imshow(X_train[0, :, :])
    # plt.show()
    # input("hi")
    #######################
    return X_train, X_test, y_train, y_test

def read_CIFAR10_dataset(n_samples_train=None, n_samples_test=None):
    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data?version=stable
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #######################
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    #######################
    if n_samples_train is not None:
        X_train = X_train[:n_samples_train, :, :]
        y_train = y_train[:n_samples_train]
    if n_samples_test is not None:
        X_test = X_test[:n_samples_test, :, :]
        y_test = y_test[:n_samples_test]
    #######################
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # plt.imshow(X_train[0, :, :, :])
    # plt.show()
    # input("hi")
    #######################
    return X_train, X_test, y_train, y_test

def separate_classes(X_train, X_test, y_train, y_test):
    unique_labels = np.unique(y_train)
    unique_labels = np.sort(unique_labels)
    n_classes = len(unique_labels)
    X_train_classes = [None] * n_classes
    y_train_classes = [None] * n_classes
    X_test_classes = [None] * n_classes
    y_test_classes = [None] * n_classes
    for class_index in range(n_classes):
        class_label = unique_labels[class_index]
        X_train_classes[class_index] = X_train[y_train == class_label, :, :]
        y_train_classes[class_index] = y_train[y_train == class_label][0]
        X_test_classes[class_index] = X_test[y_test == class_label, :, :]
        y_test_classes[class_index] = y_test[y_test == class_label][0]
    return X_train_classes, X_test_classes, y_train_classes, y_test_classes


if __name__ == "__main__":
    main()
