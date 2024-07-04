import os
import sys
import random
import datetime
import time
import shutil
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import math

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.python.keras.callbacks_v1 import TensorBoard

# Import specific modules from your project
from Model import SPP, ResUNet, Baseline
# Define command-line arguments
parser = argparse.ArgumentParser(description='DeNoiser Training')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 500)')
parser.add_argument('--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--network', default='ResUNet', type=str,
                    help='network')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate (default: 5e-4)')
parser.add_argument('--scheduler', default='constant-lr', type=str,
                    help='scheduler')
parser.add_argument('--batch-norm', action='store_true',
                    help='apply batch norm')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default='0', type=str,
                    help='GPU id(s) to use. Separate multiple ids by comma.')
parser.add_argument('--base-lr', '--base-learning-rate', default=5e-6, type=float,
                    help='base learning rate (default: 5e-6)')

def main():
    args = parser.parse_args()
    check_gpu()  # Check if TensorFlow is using GPU
    main_worker(args)

# Worker function
def main_worker(args):
    # Set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load dataset
    data = "./data/"

    from sklearn.model_selection import train_test_split

    if args.network == "detect":
        Input_Data = os.path.join(data, 'dataset_detect_01104.npy')
        Output_Data = os.path.join(data, 'labels_detect_01104.npy')
        Inputs = np.load(Input_Data)
        Input = Inputs.reshape(-1, Inputs.shape[2], Inputs.shape[3], Inputs.shape[4])
        Output = np.load(Output_Data)
        Output = Output.reshape(-1, 2)
        print(Output.shape)
        print("start training...")
        input_train, input_val, output_train, output_val = train_test_split(Input, Output, test_size=0.3, random_state=42)
    else:
        Input_Data = os.path.join(data, 'dataset_noise_01104.npy')
        Inputs = np.load(Input_Data)
        Inputs = Inputs.reshape(-1, Inputs.shape[2], Inputs.shape[3], Inputs.shape[4])
        Input = Inputs[:, 1, :, :]
        Output = Inputs[:, 0, :, :]
        input_train, input_val, output_train, output_val = train_test_split(Input, Output, test_size=0.2, random_state=42)

    # Phân chia dữ liệu thành tập huấn luyện và tập validation

    print("----------------------------------")
    # In thông tin về kích thước của các tập dữ liệu
    print("Số lượng mẫu trong tập huấn luyện:", len(input_train))
    print("Số lượng mẫu trong tập validation:", len(input_val))

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((input_train, output_train)).batch(args.batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((input_val, output_val)).batch(args.batch_size)

    # Define criterion(s) and scheduler(s)
    criterion = tf.keras.losses.MeanAbsoluteError()
    criterion_MSE = tf.keras.losses.MeanSquaredError()
    criterion_BNR = tf.keras.losses.BinaryCrossentropy()

    if args.scheduler == "decay-lr":
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.lr,
            decay_steps=50,
            decay_rate=0.2,
            staircase=False
        )
    else:  # constant-lr
        scheduler = args.lr

    # Define optimizer
    if args.optimizer == "sgd":
        optimizer = optimizers.SGD(learning_rate=scheduler)
    elif args.optimizer == "adamW":
        optimizer = optimizers.AdamW(learning_rate=scheduler)
    else:  # Adam
        optimizer = optimizers.Adam(learning_rate=scheduler)

    print('Started Training')
    print('Training Details:')
    print('Network:         {}'.format(args.network))
    print('Epochs:          {}'.format(args.epochs))
    print('Batch Size:      {}'.format(args.batch_size))
    print('Optimizer:       {}'.format(args.optimizer))
    print('Scheduler:       {}'.format(args.scheduler))
    print('Learning Rate:   {}'.format(args.lr))

    # Define log directory and model save path
    DATE = datetime.datetime.now().strftime("%Y_%m_%d")
    log_dir = "runs/{}_{}_{}_{}".format(DATE, args.optimizer, args.scheduler, args.network)
    os.makedirs(log_dir, exist_ok=True)
    models_dir = "{}/{}_{}_{}_{}.weights.h5".format(log_dir, DATE, args.optimizer, args.scheduler, args.network)


    # Create TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    if args.network == "detect":
        model = SPP.SSPmodel()
        model.compile(optimizer=optimizer, loss=criterion_BNR, metrics=['accuracy'])
        model.summary()
        history = model.fit(train_dataset, validation_data= val_dataset, epochs=args.epochs)
    else:
        model = ResUNet.ResUNet_model()
        model.compile(optimizer = optimizer, loss = criterion, metrics=['mse'])
        model.summary()
        history = model.fit(train_dataset, validation_data= val_dataset, epochs=args.epochs)


    # learning curves
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    if args.network == "detect":
        ax.set_ylabel('Accuracy', size=15)
    else:
        ax.set_ylabel('MSE', size=15)
    ax.set_xlabel('Epoch', size=15)
    if args.network == "detect":
        lns1 = plt.plot(history.history['accuracy'], label='Acc_training', color='r')
        lns2 = plt.plot(history.history['val_accuracy'], label='Acc_validation', color='g')

        ax2 = ax.twinx()
        ax2.set_ylabel('Loss', size=15)
        lns3 = plt.plot(history.history['loss'], label='Loss_training', color='b')
        lns4 = plt.plot(history.history['val_loss'], label='Loss_validation', color='orange')
        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=7)
        plt.savefig('./plot_detect.png'.format(log_dir))  # Lưu hình ảnh dưới dạng tệp PNG
        plt.show()
    else:
        lns1 = plt.plot(history.history['mse'], label='MSE', color='b')
        lns2 = plt.plot(history.history['val_mse'], label='MSE_validation', color='r')

        ax2 = ax.twinx()
        ax2.set_ylabel('Loss', size=15)
        lns3 = plt.plot(history.history['loss'], label='Loss_training', color='b')
        lns4 = plt.plot(history.history['val_loss'], label='Loss_validation', color='orange')
        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=7)
        plt.savefig('./plot_model.png'.format(log_dir))  # Lưu hình ảnh dưới dạng tệp PNG
        plt.show()
    # Save trained model

    model.save_weights(models_dir)
    print(models_dir)
    print('Finished Training')
def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # GPUs are available
        print("Number of GPUs Available:", len(gpus))
        for gpu in gpus:
            print("GPU Name:", gpu.name)
    else:
        print("No GPUs Available. TensorFlow will use CPU.")

if __name__ == '__main__':
    check_gpu()
    main()
