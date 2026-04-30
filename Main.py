import numpy as np
import os
import cv2 as cv
from numpy import matlib
from AGTO import AGTO
from AVOA import AVOA
from BWO import BWO
from Global_Vars import Global_Vars
from LO import LO
from Model_3D_Yolov7 import Model_3D_Yolov7
from Model_HC_MRAN import Model_HC_MRAN
from Model_LSTM import Model_LSTM
from Model_Mobilenet import Model_Mobilenet
from Model_Resnet import Model_Resnet
from Model_VGG16 import Model_VGG16
from Model_Yolov3 import Model_Yolov3
from Model_Yolov5 import Model_Yolov5
from Model_Yolov7 import Model_Yolov7
from Obj_Seg import Obj_Seg, objfun
from PROPOSED import PROPOSED
from Plot_Results import *
from kalman import kalman


# Read the dataset
an = 0
if an == 1:
    Dataset = './Dataset/Dataset_1/Real Life Violence Dataset/'
    path = os.listdir(Dataset)
    filename_1 = '/Images/NonViolence/'
    filename_2 = '/Images/Violence/'
    Images = []
    Targets = []
    for i in range(len(path)):
        file = Dataset + path[i]
        File_path = os.listdir(file)
        for j in range(len(File_path)):
            print(i, j)
            img_file = file + '/' + File_path[j]
            tar = img_file.split('/')[-2]
            if tar == 'NonViolence':
                tar = 0
            else:
                tar = 1
            capture = cv.VideoCapture(img_file)
            currentframe = 0
            while True:
                ret, frame = capture.read()
                if ret:
                    # if len(frame.shape) == 3:
                    #     frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                    frame = np.uint8(frame)
                    frame = cv.resize(frame, (512, 512))
                    Images.append(frame)
                    Targets.append(tar)
                    currentframe += 1
                else:
                    break
            capture.release()

    Images = np.asarray(Images)
    Targets = np.asarray(Targets).reshape(-1,1)
    np.save('Images.npy', Images)
    np.save('Targets.npy', Targets)


# Optimization for Classification
an = 0
if an == 1:
    Feat = np.load('Images.npy', allow_pickle=True)  # Load the Preprocess
    Target = np.load('Targets.npy', allow_pickle=True)  # Load the Target
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Accelaration, Difference, Uncertainty in the measurement (Observation Error)
    xmin = matlib.repmat(np.asarray([1, 1, 20]), Npop, 1)
    xmax = matlib.repmat(np.asarray([10, 10, 100]), Npop, 1)
    fname = objfun
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("BWO...")
    [bestfit1, fitness1, bestsol1, time1] = BWO(initsol, fname, xmin, xmax, Max_iter)  # BWO

    print("AVOA...")
    [bestfit2, fitness2, bestsol2, time2] = AVOA(initsol, fname, xmin, xmax, Max_iter)  # AVOA

    print("LO...")
    [bestfit3, fitness3, bestsol3, time3] = LO(initsol, fname, xmin, xmax, Max_iter)  # LO

    print("AGTO...")
    [bestfit4, fitness4, bestsol4, time4] = AGTO(initsol, fname, xmin, xmax, Max_iter)  # AGTO

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                   bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol_Prep.npy', np.asarray(BestSol))  # Best solution


# Preprocess
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)  # Load the Images
    BestSol = np.load('BestSol_Prep.npy', allow_pickle=True)  # Load the Images
    Preprocess = kalman(Images, sol=BestSol[-1, :])
    np.save('Preprocess.npy', Preprocess)


# Classification by Varying Activation
an = 0
if an == 1:
    Feat = np.load('Preprocess.npy', allow_pickle=True)  # Load the Preprocess
    Target = np.load('Targets.npy', allow_pickle=True)  # Load the Target
    learning_percenatge = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    Batch_Size = [4, 16, 32, 64, 128]
    EVAL = []
    for BS in range(len(Batch_Size)):
        Eval = np.zeros((10, 25))
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval[0, :], pred1 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
        Eval[1, :], pred2 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
        Eval[2, :], pred3 = Model_Mobilenet(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
        Eval[3, :], pred4 = Model_Resnet(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
        Eval[4, :], pred5 = Model_HC_MRAN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
        EVAL.append(Eval)
    np.save('Eval_all.npy', np.asarray(EVAL))  # Save the Eval all


# Optimization
an = 0
if an == 1:
    Feat = np.load('Preprocess.npy', allow_pickle=True)  # Load the Datas
    Global_Vars.Feat = Feat
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Learning rate, Epoch in  LLaDA-ASIGLT
    xmin = matlib.repmat(np.asarray([5, 0.01, 5]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 0.99, 50]), Npop, 1)
    fname = Obj_Seg
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("BWO...")
    [bestfit1, fitness1, bestsol1, time1] = BWO(initsol, fname, xmin, xmax, Max_iter)  # BWO

    print("AVOA...")
    [bestfit2, fitness2, bestsol2, time2] = AVOA(initsol, fname, xmin, xmax, Max_iter)  # AVOA

    print("LO...")
    [bestfit3, fitness3, bestsol3, time3] = LO(initsol, fname, xmin, xmax, Max_iter)  # LO

    print("AGTO...")
    [bestfit4, fitness4, bestsol4, time4] = AGTO(initsol, fname, xmin, xmax, Max_iter)  # AGTO

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol_Seg = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                   bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness_Seg.npy', np.asarray(fitness))
    np.save('BestSol_Seg.npy', np.asarray(BestSol_Seg))  # Bestsol classification


# Segmentation
an = 0
if an == 1:
    Images = np.load('Preprocess.npy', allow_pickle=True)  # Load the Datas
    BestSol = np.load('BestSol.npy', allow_pickle=True)  # Load the BestSol
    Eval = np.zeros((10, 15))
    for j in range(BestSol.shape[0]):
        print(j)
        sol = BestSol[j, :]
        Model_0, Eval[j, :] = Model_3D_Yolov7(Images, sol=sol)
    Model_1, Eval[5, :] = Model_Yolov3(Images)
    Model_2, Eval[6, :] = Model_Yolov5(Images)
    Model_3, Eval[7, :] = Model_Yolov7(Images)
    Model_4, Eval[8, :] = Model_3D_Yolov7(Images)
    Proposed, Eval[9, :] = Model_3D_Yolov7(Images, sol=BestSol[4, :])
    Seg = [Model_1, Model_2, Model_3, Model_4, Proposed]
    np.save('Seg_img.npy', Seg)  # save the Segmented images of all the models
    np.save('Segmented_image.npy', Proposed)  # save the Segmented image of the Proposed model


plot_Con_results()
PLot_ROC()
Plot_Confusion()
plot_results_Para()
plot_results_detect()
Sample_Image_Results()
Image_Results()
