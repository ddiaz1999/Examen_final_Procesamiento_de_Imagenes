import numpy as np
import os
import cv2
from hough import*
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

class bandera:

    def __init__(self,path_file):
        self.image = cv2.imread(path_file)

    def colores(self):
        print('calculating number of colors...')
        number_max_colors = 4
        image_to_process = np.array(self.image, dtype=np.float64) / 255
        rows, cols, ch = image_to_process.shape
        image_array = np.reshape(image_to_process, (rows * cols, ch))
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        model = KMeans(n_clusters=number_max_colors, random_state=0).fit(image_array_sample)
        self.labels = model.predict(image_array)

        number_of_colors = np.max(self.labels)+1
        print(f'number of colors is {number_of_colors}')
        print("")
        return number_of_colors

    def porcentaje(self):
        print('calculating percetage of each color...')
        label_index = np.arange(0, len(np.unique(self.labels)) + 1)
        (histogram, _) = np.histogram(self.labels, bins = label_index)
        histogram = histogram.astype('float')
        histogram /= histogram.sum()

        histogram = histogram*100
        print(f'percetage of each color = {histogram}')
        print("")
        return histogram

    def orientacion(self):
        orientation = ['vertical','horizontal','mixed']
        print('calculating orientation of the image...')
        high_thresh = 300
        bw_edges = cv2.Canny(self.image, high_thresh * 0.3, high_thresh, L2gradient=True)
        hough_tf = hough(bw_edges)
        accumulator = hough_tf.standard_HT()

        acc_thresh = 50
        N_peaks = 11
        nhood = [25, 9]
        peaks = hough_tf.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        thetas = []
        for i in range(len(peaks)):
            theta_ = hough_tf.theta[peaks[i][1]]
            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            thetas.append(theta_)

        horizontal = 0
        vertical = 0
        zeros = 0

        for i in range(len(thetas)):
            if 83<np.abs(thetas[i])<92:
                horizontal += 1
            elif 175<np.abs(thetas[i])<185:
                vertical += 1
            elif thetas[i] == 0:
                zeros += 1

        if horizontal >= 2:
            if zeros >= 2:
                print(f'orientation of image is: {orientation[2]}')
                return orientation[2]
            else:
                print(f'orientation of image is: {orientation[1]}')
                return orientation[1]

        if vertical >= 2:
            print(f'orientation of image is: {orientation[0]}')
            return orientation[0]