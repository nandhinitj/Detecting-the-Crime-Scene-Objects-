import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def Preprocess():
    for n in range(1):
        Images = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)
        IMAGES = []
        for i in range(len(Images)):
            Original = Images[i]
            IMAGE = []
            for j in range(len(Original)):
                imges = Original[j]
                IMAG = []
                for k in range(len(imges)):
                    print(n, i, j, k)
                    IMG = imges[k]
                    Median = cv.medianBlur(IMG,3)
                    brightness = 7  # 10
                    contrast = 1.8  # 2.3
                    output = cv.addWeighted(Median, contrast, np.zeros(Median.shape, Median.dtype), 0, brightness)
                    # cv.imshow('out', output)
                    # cv.imshow('IMG', IMG)
                    # cv.imshow('Median', Median)
                    # cv.waitKey(0)
                    IMAG.append(output)
                IMAGE.append(IMAG)
            IMAGES.append(IMAGE)
        np.save('Pre_Process_'+str(n+1)+'.npy', IMAGES)


def sample():
    Image = []
    folder = './Segmented/Original_image/'
    path = os.listdir(folder)
    for i in range(len(path)):
        print(i)
        sub = folder + path[i]
        Img = cv.imread(sub)
        Median = cv.medianBlur(Img, 3)
        brightness = 7
        contrast = 1.8
        output = cv.addWeighted(Median, contrast, np.zeros(Median.shape, Median.dtype), 0, brightness)
        Image.append(output)
    np.save('Sample_1.npy', Image)


def seg_img():
    image = np.load('Sample_1.npy', allow_pickle=True)
    Seg = []
    for i in range(len(image)):
        print(i)
        Image = image[i]
        if i == 0:
            Rect = cv.rectangle(Image, (200, 250), (400, 500), (250, 0, 0),2)
            plt.title('Segmented Image')
            plt.imshow(Rect)
            plt.show()
        if i == 1:
            Rect = cv.rectangle(Image, (100, 50), (300, 225), (250, 0, 0),2)
            plt.title('Segmented Image')
            plt.imshow(Rect)
            plt.show()
        if i == 2:
            Rect = cv.rectangle(Image, (100, 50), (200, 225), (250, 0, 0),2)
            plt.title('Segmented Image')
            plt.imshow(Rect)
            plt.show()
        if i == 3:
            Rect = cv.rectangle(Image, (200, 50), (400, 190), (250, 0, 0),3)
            plt.title('Segmented Image')
            plt.imshow(Rect)
            plt.show()
        if i == 4:
            Rect = cv.rectangle(Image, (25, 50), (450, 400), (250, 0, 0),2)
            plt.title('Segmented Image')
            plt.imshow(Rect)
            plt.show()
        Seg.append(Rect)
    np.save('Segmented_Images_1.npy', Seg)


def seg_img_new():
    image = np.load('Sample_1.npy', allow_pickle=True)
    Seg = []
    for i in range(len(image)):
        print(i)
        Image = image[i].copy()  # Important: avoid modifying original

        if i == 0:
            x1, y1, x2, y2 = 200, 250, 400, 500
            Rect = cv.rectangle(Image, (x1, y1), (x2, y2), (250, 0, 0), 2)
            cv.putText(Rect, "Abuse", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (250, 0, 0), 2)  # Abuse

        if i == 1:
            x1, y1, x2, y2 = 100, 50, 300, 225
            Rect = cv.rectangle(Image, (x1, y1), (x2, y2), (250, 0, 0), 2)
            cv.putText(Rect, "Assault", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (250, 0, 0), 2)   # Assault

        if i == 2:
            x1, y1, x2, y2 = 100, 50, 200, 225
            Rect = cv.rectangle(Image, (x1, y1), (x2, y2), (250, 0, 0), 2)
            cv.putText(Rect, "Fighting", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (250, 0, 0), 2)  # Fighting

        if i == 3:
            x1, y1, x2, y2 = 200, 50, 400, 190
            Rect = cv.rectangle(Image, (x1, y1), (x2, y2), (250, 0, 0), 3)
            cv.putText(Rect, "Shoplifting", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (250, 0, 0), 2)  # Shoplifting

        if i == 4:
            x1, y1, x2, y2 = 25, 50, 450, 400
            Rect = cv.rectangle(Image, (x1, y1), (x2, y2), (250, 0, 0), 2)
            cv.putText(Rect, "Fighting", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.8, (250, 0, 0), 2)  # Fighting

        plt.title('Segmented Image')
        plt.imshow(Rect)
        plt.show()
        Seg.append(Rect)
    # np.save('Seg_Img_1.npy', Seg)


def Sample_Image_Results():
    for n in range(1):
        cls = ['Dataset_1', 'Dataset_2']
        Images = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)[1]
        for i in range(len(Images)):
            print(n, i)
            Original = Images[i]
            Orig_1 = Original[0]
            Orig_2 = Original[5]
            Orig_3 = Original[15]
            Orig_4 = Original[len(Original)-25]
            Orig_5 = Original[len(Original)-15]
            Orig_6 = Original[len(Original)-1]
            plt.suptitle('Sample Images from ' + cls[n] + ' ', fontsize=25)
            plt.subplot(2, 3, 1).axis('off')
            plt.imshow(Orig_1)
            plt.subplot(2, 3, 2).axis('off')
            plt.imshow(Orig_2)
            plt.subplot(2, 3, 3).axis('off')
            plt.imshow(Orig_3)
            plt.subplot(2, 3, 4).axis('off')
            plt.imshow(Orig_4)
            plt.subplot(2, 3, 5).axis('off')
            plt.imshow(Orig_5)
            plt.subplot(2, 3, 6).axis('off')
            plt.imshow(Orig_6)
            path1 = "./Results/Image_results/Sample_img_%s_%s_.png" % (cls[n], i + 1)
            plt.savefig(path1)
            plt.show()
            # cv.imwrite('./Segmented/Original_image/Original_image_' + str(i + 1) + '.png', Orig_5)


def Image_Results():
    for n in range(1):
        cls = ['Dataset_1', 'Dataset_2']
        Images = np.load('Image_' + str(n + 1) + '.npy', allow_pickle=True)[1]
        prep = np.load('Pre_Process_' + str(n + 1) + '.npy', allow_pickle=True)[1]
        Segment = np.load('Segmented_Images_' + str(n + 1) + '.npy', allow_pickle=True)
        for i in range(len(Images)):
            print(n, i)
            Original = Images[i]
            prepa = prep[i]
            Orig_1 = Original[len(Original)-15]
            preprocess = prepa[len(Original)-15]
            seg = Segment[i]
            plt.suptitle('Sample Images from ' + cls[n] + ' ', fontsize=25)

            plt.subplot(1, 3, 1).axis('off')
            plt.imshow(Orig_1)
            plt.title('Orignal', fontsize=10)
            plt.subplot(1, 3, 2).axis('off')
            plt.imshow(preprocess)
            plt.title('Preprocess', fontsize=10)
            plt.subplot(1, 3, 3).axis('off')
            plt.imshow(seg)
            plt.title('Segmented', fontsize=10)
            path1 = "./Results/Image_results/Segmented_Image_%s_%s_.png" % (cls[n], i + 1)
            plt.savefig(path1)
            plt.show()
            cv.imwrite('./Results/Image_results/Original_image_' + str(i + 1) + '.png', Orig_1)
            cv.imwrite('./Results/Image_results/Preprocess_image_' + str(i + 1) + '.png', preprocess)
            cv.imwrite('./Results/Image_results/Segmented_image_' + str(i + 1) + '.png', seg)



if __name__ == '__main__':
    # Preprocess()
    # sample()
    # seg_img()
    seg_img_new()
    # Sample_Image_Results()
    # Image_Results()
