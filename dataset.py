import numpy as np
import cv2
from scipy import io


def load_hoda(training_Sample_Size=1000, test_sample_size=200, size=5):
    # بارگذاری دیتاست
    dataset = io.loadmat('./dataset/Data_hoda_full.mat')

    # استخراج تصاویر از ستون اول (هر عنصر یک تصویر است)
    x_train_original = dataset['Data'][:training_Sample_Size, 0]  # ← کلید اصلی
    x_test_original = dataset['Data'][training_Sample_Size:training_Sample_Size + test_sample_size, 0]

    # برچسب‌ها
    y_train = np.squeeze(dataset['labels'][:training_Sample_Size])
    y_test = np.squeeze(dataset['labels'][training_Sample_Size:training_Sample_Size + test_sample_size])

    # تغییر سایز به ۵×۵
    x_train_5by5 = [cv2.resize(img, dsize=(size, size)) for img in x_train_original]
    x_test_5by5 = [cv2.resize(img, dsize=(size, size)) for img in x_test_original]

    # تغییر شکل به بردار
    x_train = np.reshape(x_train_5by5, [-1, size ** 2])
    x_test = np.reshape(x_test_5by5, [-1, size ** 2])

    return x_train, y_train, x_test, y_test