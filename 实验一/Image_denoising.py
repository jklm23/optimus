import numpy as np
import imageio
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import warnings
import os

warnings.filterwarnings("ignore")


def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.double) / info.max


def write(im, filename):
    img = np.copy(im)
    img = img.squeeze()
    if img.dtype == np.double:
        #img = np.array(img*255, dtype=np.uint8)
        img = img * np.iinfo(np.uint8).max
        img = img.astype(np.uint8)
    imageio.imwrite(filename, img)


# get the noise mask of corrImg
def getNoiseMask(corrImg):
    return np.array(corrImg != 0, dtype='double')


def restore(img, filename):
    radius = 4
    resImg = np.copy(img)
    noiseMask = getNoiseMask(img)
    rows, cols, channel = img.shape
    count = 0
    for row in range(rows):
        for col in range(cols):

            if row-radius < 0:
                rowl = 0
                rowr = rowl+2*radius
            elif row+radius >= rows:
                rowr = rows-1
                rowl = rowr-2*radius
            else:
                rowl = row-radius
                rowr = row+radius

            if col-radius < 0:
                coll = 0
                colr = coll+2*radius
            elif col+radius >= cols:
                colr = cols-1
                coll = colr-2*radius
            else:
                coll = col-radius
                colr = col+radius

            for chan in range(channel):
                if noiseMask[row, col, chan] != 0.:
                    continue
                x_train = []
                y_train = []
                for i in range(rowl, rowr):
                    for j in range(coll, colr):
                        if noiseMask[i, j, chan] == 0.:
                            continue
                        if i == row and j == col:
                            continue
                        x_train.append([i, j])
                        y_train.append([img[i, j, chan]])
                if x_train == []:
                    continue
                Regression = Lasso(alpha=0.5)
                Regression.fit(x_train, y_train)
                resImg[row, col, chan] = Regression.predict([[row, col]])
            count += 1
            if count % 50000 == 0:
                print(filename+" restored:" + str(float(count)/rows/cols))
    print(filename+" restore finish!")
    return resImg

if __name__ == '__main__':
    os.chdir('/root/mytuyouhua/实验一')
    noiseRatio = 0.6
    
    Img = im2double(imageio.imread('./mypicture.png'))
    Img[(Img == 0)] = 0.01
    rows, cols, channels = Img.shape

    # 加随机噪声
    noiseMask = np.ones((rows, cols, channels))
    subNoiseNum = round(noiseRatio * cols)
    for k in range(channels):
        for i in range(rows):
            tmp = np.random.permutation(cols)
            noiseIdx = np.array(tmp[:subNoiseNum])
            noiseMask[i, noiseIdx, k] = 0
    noiseImg = Img * noiseMask
    write(im=noiseImg, filename='mypicture_noise.png')

    ##  重建
    img = im2double(imageio.imread('mypicture_noise.png'))
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    resImg = restore(img, 'mypicture_rebuild.png')
    write(resImg, 'mypicture_rebuild.png')
