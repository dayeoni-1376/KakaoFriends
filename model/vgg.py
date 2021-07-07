import tensorflow as tf 
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras import layers, models, regularizers


class VGG():

    def __init__(self, vgg_type = 16, base_trainable = True):
        super(VGG, self).__init__()
        vgg_spec = {16: VGG16(input_shape = (224, 224, 3) , include_top = False, weights = 'imagenet'), 
                    19: VGG19(input_shape = (224, 224, 3) , include_top = False, weights = 'imagenet')}
        self.base_model = vgg_spec[vgg_type]
        self.base_model.trainable = base_trainable # True / False 

        self.global_average_layer = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()

        # Dense layer에 kernel_regularizer를 추가하면 성능이 좀 더 빨리 올라감. 
        self.fc_1 = layers.Dense(4096, 
                                    kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 =0.001), 
                                    activation = 'relu')
        self.fc_2 = layers.Dense(2048, 
                                    kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 =0.001),
                                    activation = 'relu')
        self.fc_3 = layers.Dense(1024,
                                    kernel_regularizer = regularizers.l1_l2(l1 = 0.001, l2 =0.001),
                                    activation = 'relu')
        self.fc_4 = layers.Dense(7, activation = 'sigmoid')
        self.dropout_1 = layers.Dropout(0.5)
        self.dropout_2 = layers.Dropout(0.5)

    def build(self):
        x = tf.keras.Sequential()
        x.add(self.base_model)

        x.add(self.flatten)
        x.add(self.fc_1)
        x.add(self.dropout_1)
        x.add(self.fc_2)
        x.add(self.dropout_2)
        x.add(self.fc_3)
        x.add(self.fc_4)
        return x



model = VGG()
# model = vgg.build()
print(model.summary())