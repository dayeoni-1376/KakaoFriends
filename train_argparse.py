import argparse

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dataset import read_dataframe, make_imageGenerator
from model.vgg import VGG
"""
- version 
    python      v3.8 
    tensorflow  v2.3 
"""

BATCHSIZE = 32
SEED = 42
LEARNINGRATE = 0.001

def parse_arg(args):
    parser = argparse.ArgumentParser(description = 'transfer learning scripts for using VGG')
    parser.add_argument('--model_case', default = 1)
    parser.add_argument('--epochs', default = 400, type = int)
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--checkpoints-dir', default = '.\\result\\', help = 'Directory to store checkpoints of model during training')

    #

    return parser.parse_args(args)

def main(args):
    # 데이터 준비
    train_df, val_df, test_df = read_dataframe()
    train_generator , val_generator, test_generator = make_imageGenerator(train_df, val_df, test_df, 
                                                                          BATCHSIZE = BATCHSIZE, 
                                                                          SEED = SEED)
    # tensorflow GPU memory 사용 제한 코드
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # model compile and training 
    model = VGG().build()

    opt = SGD(lr = LEARNINGRATE, decay = 1e-6, momentum = 0.0, nesterov = True)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


    # callback
    lr_reduction = ReduceLROnPlateau(
        monitor = 'val_loss', 
        patience = 3, 
        factor = 0.5,
        min_lr = 0.0001,
        verbose = 1
    )
    earlystop = EarlyStopping(patience = 50)

    cp_path = f'.\\result\\training\\cp_model{model_case}.ckpt'


if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])
    main(args)
