'''
제목: Korean-4 CSV 데이터 증식, 파일로 저장
기능: Korean-4 데이터를 증식하고 CSV 파일로 저장
날짜: 2023년 12월 23일 - 2023년 12월 24일
작성: 남상혁 (한서대 항공컴퓨터전공 201902583 nsh142@naver.com)

사용법
    DataSaveCsv(Korean-4 데이터 파일명 리스트,
                Korean-4 데이터 파일 경로,
                Korean-4 데이터를 저장할 파일명,
                Korean-4.csv를 저장할 경로,
                데이터를 증식할 갯수)
                
    저장 경로에 증식된 Korean-4 데이터셋 파일명.csv로 저장됨.
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

class DataAugSaveCsv():
    def __init__(self, file_name_list, file_path, save_file_name, save_file_path = '', augmentation_num = 0):
        self.file_path = file_path
        self.file_name_list = file_name_list
        self.augmentation_num = augmentation_num - 1
        self.data_save_to_csv(save_file_name, save_file_path)

    def set_image_data(self, image):
        for pixel in range(self.image_len):
            self.batch_dict[pixel + 1].append(image[pixel])

    def image_reshape(self, image):
        reshaped_image = image.reshape(3072)
        return reshaped_image

    def image_augmentation(self, image):
        image_list = [image, ]
        data = img_to_array(image)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(
                rotation_range = 30,
                width_shift_range = 0.2,
                height_shift_range = 0.1,
                shear_range = 0.9,
                zoom_range = 0.2,
                fill_mode='nearest')
        it = datagen.flow(samples, batch_size=1)

        for i in range(self.augmentation_num):
            batch = it.next()
            image = batch[0]
            image_list.append(image)
        return image_list

    def image_resize(self, image):
        resized_image = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
        return resized_image

    def set_data_batch(self):
        percent = len(self.file_name_list) // 10
        per_count = 0
        per = 0
        
        for file_name in self.file_name_list:
            load_image = self.file_path + file_name
            image = plt.imread(load_image)
            image = image.astype('uint8')
            resized_image = self.image_resize(image)
            image_list = self.image_augmentation(resized_image)
            for img in image_list:
                reshaped_image = self.image_reshape(img)
                self.set_image_data(reshaped_image)
                self.set_label_batch(file_name)
            if(per_count % percent == 0):
                print(per, "%", end = '->')
                per += 10
            per_count += 1

    def create_batch_dict(self): 
        self.image_len = 3072
        for pixel in range(self.image_len + 1):
            self.batch_dict[pixel] = []       

    def set_label_batch(self, file_name):
        label_dict = {'ㄱ':0, 'ㄴ':1, 'ㄷ':2, 'ㄹ':3}
        korean = file_name[0]
        if(korean in label_dict.keys()):
            label = label_dict[korean]
            self.batch_dict[0].append(label)

    def set_batch_dict(self):
        keys = range(0, 3073)
        self.batch_dict = dict.fromkeys(keys)
        self.create_batch_dict()
        self.set_data_batch()

    def data_to_DataFrame(self):
        self.set_batch_dict()
        df = pd.DataFrame(self.batch_dict)
        return df

    def dataFrame_shuffle(self, df):
        df_shuffled=df.iloc[np.random.permutation(df.index)].reset_index(drop=True)
        return df_shuffled
    
    def data_save_to_csv(self, save_file_name, save_file_path = ''):
        df = self.data_to_DataFrame()
        df_shuffled = self.dataFrame_shuffle(df)
        file = save_file_path + save_file_name + ".csv"
        df_shuffled.to_csv(file, index = False)
        print("done!")
