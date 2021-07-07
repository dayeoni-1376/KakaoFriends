import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


from dataset import read_dataframe, make_imageGenerator
from model.vgg import VGG
"""
- version 
    python      v3.8 
    tensorflow  v2.3 
"""
MODEL_CASE = 1
ADDITIONAL_LEARNING = False
EPOCHS = 400


BATCHSIZE = 32
SEED = 42
LEARNINGRATE = 0.001



if __name__ == '__main__':
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

    cp_path = f'.\\results\\training_cp\\cp_model{MODEL_CASE}.ckpt'
    cp_dir = os.path.dirname(cp_path)
    cp_callback = ModelCheckpoint(filepath = cp_path, save_weights_only = True, verbose = 1)
    callbacks = [earlystop, lr_reduction, cp_callback]

    if ADDITIONAL_LEARNING == False:
        history = model.fit_generator(train_generator, 
                                    validation_data = val_generator, 
                                    epochs = EPOCHS, 
                                    callbacks = [callbacks])
        np.save('{MODEL_CASE}_model.npy', history.history)    
        model.save(f'.\\results\\output_model\\{MODEL_CASE}_model.h5')
    

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{MODEL_CASE}_model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc = 'upper left')
        plt.savefig(f'.\\results\\plot\\{MODEL_CASE}_accuracy.png')

        # sumarize history for loss 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{MODEL_CASE}_model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], los = 'upper left')
        plt.savefig(f'.\\results\\plot\\{MODEL_CASE}_loss.png')

    else:
        tf.keras.models.load_model(f'.\\results\\output_model\\{MODEL_CASE}_model.h5')
        history_new = model.fit_generator(train_generator, 
                            validation_data = val_generator, 
                            epochs = EPOCHS, 
                            callbacks = [callbacks])
        history_before = np.load('{MODEL_CASE}_model.npy', allow_pickle = 'TRUE').item()
        history = np.vstack([history_before, history_new])

        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{MODEL_CASE}_model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc = 'upper left')
        plt.savefig(f'.\\results\\plot\\{MODEL_CASE}_add_accuracy.png')

        # sumarize history for loss 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{MODEL_CASE}_model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], los = 'upper left')
        plt.savefig(f'.\\results\\plot\\{MODEL_CASE}_add_loss.png')