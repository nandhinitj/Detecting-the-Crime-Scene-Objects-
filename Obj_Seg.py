import numpy as np
from Evaluation import Enhanced_evaluation
from Global_Vars import Global_Vars
from Model_3D_Yolov7 import Model_3D_Yolov7
from kalman import kalman


def objfun(Soln):
    Feat = Global_Vars.Feat
    Target = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Preprocess = kalman(Feat, sol=sol)
            EVAl = []
            for img in range(len(Preprocess)):
                Eval = Enhanced_evaluation(Feat[img], Preprocess[img])
                EVAl.append(Eval)
            mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
            Fitn[i] = 1 / (mean_EVAl[0, 14] + mean_EVAl[0, 13] + mean_EVAl[0, 15])  # (Brisque + NIQE + PIQE)
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Preprocess = kalman(Feat, sol=sol)
        EVAl = []
        for img in range(len(Preprocess)):
            Eval = Enhanced_evaluation(Feat[img], Preprocess[img])
            EVAl.append(Eval)
        mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
        Fitn = 1 / (mean_EVAl[0, 14] + mean_EVAl[0, 13] + mean_EVAl[0, 15])  # (Brisque + NIQE + PIQE)
        return Fitn


def Obj_Seg(Soln):
    Images = Global_Vars.Images
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Predict, Eval = Model_3D_Yolov7(Images, sol=sol)
            Fitn[i] = 1 / Eval[0, 5]  # (Precison)
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Predict, Eval = Model_3D_Yolov7(Images, sol=sol)
        Fitn = 1 / Eval[0, 5]  # (Precison)
        return Fitn
