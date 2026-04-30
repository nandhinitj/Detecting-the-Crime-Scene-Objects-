import numpy as np
import math
from sklearn.metrics import precision_score
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from sewar.full_ref import vifp, uqi
# from sewar.no_ref import niqe


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    FPR = fp / (fp + tn) * 100
    FNR = fn / (tp + fn) * 100
    NPV = tn / (tn + fp) * 100
    FDR = fp / (tp + fp) * 100
    F1_score = (2 * tp) / (2 * tp + fp + fn) * 100
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC]
    return EVAL


def Detect_evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = sum(Tp)
    fp = sum(Fp)
    tn = sum(Tn)
    fn = sum(Fn)

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    FPR = fp / (fp + tn) * 100
    FNR = fn / (tp + fn) * 100
    NPV = tn / (tn + fp) * 100
    FDR = fp / (tp + fp) * 100
    F1_score = (2 * tp) / (2 * tp + fp + fn) * 100
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    IoU = (tp / (tp + fp + fn)) * 100
    AP = (precision_score(sp, act, average='macro')) * 100

    # [Precision, Recall] = precision_recall_at_k(sp)
    # MAP = sum(Precision.values()) / len(Precision)
    # map = (calculate_map(sp, act))

    Recall = sensitivity
    # EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC, AP, IoU]
    EVAL = [tp, tn, fp, fn, Recall, precision, AP, IoU]
    return EVAL



def MSE(img1, img2):
    return np.mean((img1 - img2) ** 2)


def RMSE(img1, img2):
    return np.sqrt(MSE(img1, img2))


def MAE(img1, img2):
    return np.mean(np.abs(img1 - img2))


def PSNR(img1, img2):
    mse = MSE(img1, img2)
    if mse == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ================= STRUCTURAL =================

def SSIM(img1, img2):
    return ssim(img1, img2, data_range=255)


def UQI(img1, img2):
    return uqi(img1, img2)


def SC(img1, img2):
    return np.sum(img1 ** 2) / (np.sum(img2 ** 2) + 1e-10)


def VIF(img1, img2):
    return vifp(img1, img2)


# ================= EDGE / FEATURE =================

def EPI(img1, img2):
    edge1 = sobel(img1)
    edge2 = sobel(img2)
    return np.sum(edge1 * edge2) / (np.sum(edge1 ** 2) + 1e-10)


def AG(img):
    gx = np.diff(img, axis=1)
    gy = np.diff(img, axis=0)
    return np.mean(np.sqrt(gx[:, :-1]**2 + gy[:-1, :]**2))


# ================= INFORMATION =================

def Entropy(img):
    return shannon_entropy(img)


def CII(original, enhanced):
    return np.std(enhanced) / (np.std(original) + 1e-10)


# ================= ADVANCED =================

def FSIM(img1, img2):
    # Convert to torch
    t1 = torch.tensor(img1 / 255.0).unsqueeze(0).unsqueeze(0).float()
    t2 = torch.tensor(img2 / 255.0).unsqueeze(0).unsqueeze(0).float()
    return float(piq.fsim(t1, t2))


# def NIQE(img):
#     return niqe(img)

# import cv2
# def NIQE(img):
#     niqe = cv2.quality.QualityNIQE_create("niqe_model.yml", "niqe_range.yml")
#     return niqe.compute(img)[0][0]

import torch
import piq

def NIQE(img):
    img = img.astype(np.float32) / 255.0
    t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return float(piq.niqe(t))


def BRISQUE(img):
    t = torch.tensor(img / 255.0).unsqueeze(0).unsqueeze(0).float()
    return float(piq.brisque(t))


# ================= PIQE (NEW) =================

def PIQE(img):
    img = img.astype(np.uint8)
    score, _, _, _ = cv2.quality.QualityPIQE_compute(img)
    return score


# ================= FULL EVALUATION =================

def Enhanced_evaluation(original, processed):
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)

    mse = MSE(original, processed)
    rmse = RMSE(original, processed)
    mae = MAE(original, processed)
    psnr = PSNR(original, processed)

    ssim_val = SSIM(original.astype(np.uint8), processed.astype(np.uint8))
    uqi_val = UQI(original, processed)
    sc = SC(original, processed)
    vif_val = VIF(original, processed)

    epi = EPI(original, processed)
    ag = AG(processed)

    entropy = Entropy(processed)
    cii = CII(original, processed)

    fsim_val = FSIM(original, processed)
    niqe_val = NIQE(processed)
    brisque_val = BRISQUE(processed)
    piqe_val = PIQE(processed)

    Values = np.asarray([
        mse, rmse, mae, psnr,
        ssim_val, uqi_val, sc, vif_val,
        epi, ag, entropy, cii,
        fsim_val, niqe_val, brisque_val, piqe_val
    ])

    return Values