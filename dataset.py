import pandas as pd 
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

from sklearn.model_selection import train_test_split

DATAPATH = '.\\data\\'


def read_dataframe():
    kakao_label = pd.read_csv(DATAPATH + 'train\\kakao_label.csv')
    test_label = pd.read_csv(DATAPATH + 'test\\test_label.csv')
    kakao_label = kakao_label.rename(columns = {'Unnamed: 0':'Filename'})
    
    train_df , val_df = train_test_split(kakao_label, shuffle = True, random_state = 42)
    test_df = test_label.rename(columns = {'Name' : 'Filename'})
    del test_df['Unnamed: 0']
    
    return train_df, val_df, test_df

def make_imageGenerator(train_df , val_df, test_df, BATCHSIZE = 32, SEED = 42):
    train_datagen = ImageDataGenerator(
        # featurewise_center=False, # input dataset의 평균을 0으로 
        # samplewise_center=False, # 표본 평균을 0으로 
        # featurewise_std_normalization=False, # input datset을 표준편차로 나누기
        # samplewise_std_normalization=False, # 표본을 표준편차로 나누기 
        # zca_whitening=False, # zca whitening 추가 
        # zca_epsilon=1e-06, # zca whitening epsilon
        rotation_range=30, # rotation 각도 범위 
        width_shift_range=0.1, # ex) 전체 넓이 100, 값이 0.1이면 10 픽셀 좌우 이동
        height_shift_range=0.1, # ex) 전체 길이 100, 값이 0.1이면 10 픽셀 상하 이동 
        # brightness_range=None, # 밝기 범위 
        # shear_range=0.0, # 밀림 강도 범위 내에서 원본 이미지 변형
        zoom_range=0.3, # 확대 혹은 축소, 값이 0.3이면 0.7배~ 1.3배 크기 변화
        # channel_shift_range=0.0, # channel 변화 범위 
        # fill_mode='nearest', 
        # cval=0.0,
        horizontal_flip=True, # 수평 방향 뒤집기
        vertical_flip=True, # 수직 방향 뒤집기
        rescale= 1./255 , # rescaling 
        # preprocessing_function=None, # 사용자 정의 preprocessing function 
        # data_format=None, 
        # validation_split=0.0, 
        # dtype=None, 
    )

    test_datagen = ImageDataGenerator(
        rescale = 1./255
    )

    KakaoFriends = ['Apeach', 'Frodo', 'JayG', 'Muzi', 'Neo', 'Ryan','Tube']
    train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = DATAPATH + 'train\\train_image\\',
        x_col = 'Filename',
        y_col = KakaoFriends,
        batch_size = BATCHSIZE,
        seed = SEED,
        shuffle = False ,
        class_mode = 'raw', 
            # multi-class case : categorical,
            # multi-label case : raw, 
            # binary-class case : binary, 
            # - input, multi_output, sparse , None 
        target_size = (224,224)
    )

    val_generator = test_datagen.flow_from_dataframe(
        dataframe = val_df, 
        directory = DATAPATH + 'train\\train_image\\',
        x_col = 'Filename',
        y_col = KakaoFriends,
        batch_size = BATCHSIZE,
        seed = SEED,
        shuffle = False,
        class_mode = 'raw',
        target_size = (224,224)
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe = test_df,
        directory = DATAPATH + 'test\\test_image\\',
        x_col = 'Filename',
        y_col = KakaoFriends, 
        batch_size = 1, 
        seed = SEED,
        shuffle = False, 
        class_mode = 'raw', 
        target_size = (224,224)
    )
    return train_generator, val_generator, test_generator


read_dataframe()