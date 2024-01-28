'''
제목: 한글 이미지를 입력받아 판별
기능: 한글 이미지 (ㄱ, ㄴ, ㄷ, ㄹ)을 입력받아 판별
날짜: 2023년 12월 23일 - 2023년 12월 24일
작성: 남상혁 (한서대 항공컴퓨터전공 201902583 nsh142@naver.com)

사용법
    Input:
        ScanKorean(한글(입력) 이미지,
                이미지 판별 모델)
                
    결과:
        한글 이미지 판별 결과
'''

import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

class ScanKorean():
    def __init__(self, input_image, model_name):
        self.load_image(input_image)
        self.image_show()
        self.load_model(model_name)
        self.image_preprocess()
        self.korean_predict()

    def load_image(self, input_image):
        image = plt.imread(input_image)
        self.image = image
    
    def image_show(self):
        print("판별할 글자 이미지")
        plt.imshow(self.image)
        plt.show()
    
    def load_model(self, model_name):
        model = keras.models.load_model(model_name)
        self.model = model

    def image_preprocess(self):
        image = self.image
        resized_image = cv2.resize(image, (32, 32),
                                    interpolation = cv2.INTER_AREA)
        array_image = np.array(resized_image)
        reshaped_image = array_image.reshape(3072)
        copy_image = []

        for pixel in reshaped_image:
            if(pixel > 0.74):
                pixel = 1
            copy_image.append(pixel)
        copy_image = np.array(copy_image)
        reshaped_clean_image = copy_image.reshape(1, 32, 32, 3)
        self.clean_image = reshaped_clean_image

    def korean_predict(self):
        korean_dict = {0:'ㄱ', 1:'ㄴ', 2:'ㄷ', 3:'ㄹ'}
        image = self.clean_image
        model = self.model
        predict = model.predict(image)
        result = np.argmax(predict, axis = -1)
        print("=" * 50)
        print("판별 결과:", korean_dict[result[0]])
        print("=" * 50)
