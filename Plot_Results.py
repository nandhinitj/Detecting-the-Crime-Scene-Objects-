import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import cv2 as cv
from itertools import cycle
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix

no_of_dataset = 1


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plot_Con_results():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'BWO-OKF-HC-MRAN', 'AVOA-OKF-HC-MRAN', 'LOA-OKF-HC-MRAN', 'AGTO-OKF-HC-MRAN',
                 'MP-AGTO-OKF-HC-MRAN']
    for i in range(Fitness.shape[0]):
        Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('------------------------------ Configuration', i + 1, 'Statistical Report ',
              '------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=2, marker='*', markerfacecolor='red',
                 markersize=5, label='BWO-OKF-HC-MRAN')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=2, marker='*', markerfacecolor='green',
                 markersize=5, label='AVOA-OKF-HC-MRAN')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=2, marker='*', markerfacecolor='blue',
                 markersize=5, label='LOA-OKF-HC-MRAN')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=2, marker='*', markerfacecolor='magenta',
                 markersize=5, label='AGTO-OKF-HC-MRAN')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=2, marker='*', markerfacecolor='black',
                 markersize=5, label='MP-AGTO-OKF-HC-MRAN')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./New_Results/Conv_%d.png" % (i + 1))
        plt.show()


def PLot_ROC():
    lw = 2
    cls = ['VGG16', 'LSTM', 'Mobilenet', 'Resnet', 'MP-AGTO-OKF-HC-MRAN']
    for a in range(no_of_dataset):  # For 5 Datasets
        Actual = np.load('Target_1.npy', allow_pickle=True).astype('int')
        colors = cycle(["blue", "crimson", "gold", "lime", "black"])  # "cornflowerblue","darkorange", "aqua"
        for i, color in zip(range(len(cls)), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./New_Results/Dataset_1_ROC_%s_.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_results_Para():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Algorithm = ['TERMS', 'BWO-OKF-HC-MRAN', 'AVOA-OKF-HC-MRAN', 'LOA-OKF-HC-MRAN', 'AGTO-OKF-HC-MRAN',
                 'MP-AGTO-OKF-HC-MRAN']
    Classifier = ['TERMS', 'VGG16', 'LSTM', 'Mobilenet', 'Resnet', 'MP-AGTO-OKF-HC-MRAN']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- Batchsize - Dataset', i + 1, 'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Batchsize - Dataset', i + 1, 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    BatchSize = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]

            plt.plot(BatchSize, Graph[0, :5], color='r', linewidth=3, marker='o', markerfacecolor='blue',
                     markersize=12,
                     label="BatchSize 4")
            plt.plot(BatchSize, Graph[1, :5], color='g', linewidth=3, marker='s', markerfacecolor='red',
                     markersize=12,
                     label="BatchSize 16")
            plt.plot(BatchSize, Graph[2, :5], color='b', linewidth=3, marker='*', markerfacecolor='green',
                     markersize=12,
                     label="BatchSize 32")
            plt.plot(BatchSize, Graph[3, :5], color='m', linewidth=3, marker='D', markerfacecolor='yellow',
                     markersize=12,
                     label="BatchSize 64")
            plt.plot(BatchSize, Graph[4, :5], color='k', linewidth=3, marker='p', markerfacecolor='cyan', markersize=12,
                     label="BatchSize 128")

            plt.xticks(BatchSize, ('BWO-OKF-HC-MRAN', 'AVOA-OKF-HC-MRAN', 'LOA-OKF-HC-MRAN', 'AGTO-OKF-HC-MRAN',
                                   'MP-AGTO-OKF-HC-MRAN'), rotation=10)

            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
            path1 = "./New_Results/BatchSize_%s_%s.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)

            ax.bar(X + 0.00, Graph[0, 5:10], color='b', edgecolor='k', width=0.10, label="BatchSize 4")
            ax.bar(X + 0.10, Graph[1, 5:10], color='#ef4026', edgecolor='k', width=0.10, label="BatchSize 16")
            ax.bar(X + 0.20, Graph[2, 5:10], color='lime', edgecolor='k', width=0.10, label="BatchSize 32")  # '#12e193'
            ax.bar(X + 0.30, Graph[3, 5:10], color='y', edgecolor='k', width=0.10,
                   label="BatchSize 64")  # color='#ff0490'
            ax.bar(X + 0.40, Graph[4, 5:10], color='k', edgecolor='k', width=0.10, label="BatchSize 128")

            plt.plot(X + 0.00, Graph[0, 5:10], 'y-o')
            plt.plot(X + 0.10, Graph[1, 5:10], 'b:s')
            plt.plot(X + 0.20, Graph[2, 5:10], 'r--d')
            plt.plot(X + 0.30, Graph[3, 5:10], 'k:s')
            plt.plot(X + 0.40, Graph[4, 5:10], 'c--d')
            plt.xticks(X + 0.10, ('VGG16', 'LSTM', 'Mobilenet', 'Resnet', 'MP-AGTO-OKF-HC-MRAN'), rotation=10)
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
            path1 = "./New_Results/BatchSize_%s_%s_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def Plot_Confusion():
    for n in range(no_of_dataset):
        Actual = np.load('Actual.npy', allow_pickle=True)
        Predict = np.load('Predict.npy', allow_pickle=True)
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[n]), np.asarray(Predict[n]))
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax)
        plt.title('Accuracy')
        path = "./New_Results/Confusion_%s.png" % (n + 1)
        plt.savefig(path)
        plt.show()


def plot_results_detect():  # For classification for precission
    eval1 = np.load('EVAL_Detect.npy', allow_pickle=True)
    Terms = ['Recall', 'precision', 'AP', 'IoU','MAP']
    Graph_Terms = [0, 1, 2, 3, 4]
    Need_Graph_Terms = [0,1,2,3,4]
    Algorithm = ['TERMS', 'BWO-A3DYv7', 'AVOA-A3DYv7', 'LOA-A3DYv7', 'AGTO-A3DYv7', 'MP-AGTO-A3DYv7']
    Classifier = ['TERMS', 'Yolo v3', 'Yolo v5', 'Yolo v7', '3D Yolov7', 'MP-AGTO-A3DYv7']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 3, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- Detection - Dataset', i + 1,
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Detection - Dataset', i + 1,
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [50, 100, 150, 200, 250]
    for i in range(eval1.shape[0]):
        Graph = np.zeros((len(Terms), eval1.shape[1], eval1.shape[2] + 1))
        for j in range(len(Terms)):
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 9:
                        Graph[j, k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[j, k, l] = eval1[i, k, l, Graph_Terms[j] + 4]

        for p in range(len(Graph_Terms)):
            for n_graph in range(len(Need_Graph_Terms)):
                if p == Need_Graph_Terms[n_graph]:
                    plt.plot(learnper, Graph[p, :, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue',
                             markersize=12,
                             label="BWO-A3DYv7")
                    plt.plot(learnper, Graph[p, :, 1], color='g', linewidth=3, marker='o', markerfacecolor='red',
                             markersize=12,
                             label="AVOA-A3DYv7")
                    plt.plot(learnper, Graph[p, :, 2], color='b', linewidth=3, marker='o', markerfacecolor='green',
                             markersize=12,
                             label="LOA-A3DYv7")
                    plt.plot(learnper, Graph[p, :, 3], color='c', linewidth=3, marker='o', markerfacecolor='yellow',
                             markersize=12,
                             label="AGTO-A3DYv7")
                    plt.plot(learnper, Graph[p, :, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan',
                             markersize=12,
                             label="MP-AGTO-A3DYv7")
                    plt.xticks(learnper, ('50', '100', '150', '200', '250'))
                    plt.xlabel('Epoch')
                    plt.ylabel(Terms[p])
                    plt.legend(loc=4)
                    plt.tight_layout()
                    path1 = "./New_Results/Detection_%s_comp_%s_%s_line_Alg.png" % (i + 1, p + 1, Terms[p])
                    plt.savefig(path1)
                    plt.show()

                    fig = plt.figure()
                    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                    X = np.arange(5)

                    ax.bar(X + 0.00, Graph[p, :, 5], color='r', width=0.10, label="Yolo v3")
                    ax.bar(X + 0.10, Graph[p, :, 6], color='g', width=0.10, label="Yolo v5")
                    ax.bar(X + 0.20, Graph[p, :, 7], color='b', width=0.10, label="Yolo v7")
                    ax.bar(X + 0.30, Graph[p, :, 8], color='m', width=0.10, label="3D Yolov7")
                    ax.bar(X + 0.40, Graph[p, :, 4], color='k', width=0.10, label="MP-AGTO-A3DYv7")
                    plt.xticks(X + 0.10, ('50', '100', '150', '200', '250'))
                    plt.xlabel('Epoch')
                    plt.ylabel(Terms[p])
                    plt.legend(loc=1)
                    path1 = "./New_Results/Detection_%s_comp_%s_%s_bar_net.png" % (i + 1, p + 1, Terms[p])
                    plt.savefig(path1)
                    plt.show()


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
            Orig_4 = Original[len(Original) - 25]
            Orig_5 = Original[len(Original) - 15]
            Orig_6 = Original[len(Original) - 1]
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
            path1 = "./New_Results/Image_results/Sample_img_%s_%s_.png" % (cls[n], i + 1)
            plt.savefig(path1)
            plt.show()
            cv.imwrite('./New_Results/Sample_Images/Sample_img_1-' + str(i + 1) + '-' + str(n + 1) + '.png', Orig_1)
            cv.imwrite('./New_Results/Sample_Images/Sample_img_2-' + str(i + 1) + '-' + str(n + 1) + '.png', Orig_2)
            cv.imwrite('./New_Results/Sample_Images/Sample_img_3-' + str(i + 1) + '-' + str(n + 1) + '.png', Orig_3)
            cv.imwrite('./New_Results/Sample_Images/Sample_img_4-' + str(i + 1) + '-' + str(n + 1) + '.png', Orig_4)
            cv.imwrite('./New_Results/Sample_Images/Sample_img_5-' + str(i + 1) + '-' + str(n + 1) + '.png', Orig_5)
            cv.imwrite('./New_Results/Sample_Images/Sample_img_6-' + str(i + 1) + '-' + str(n + 1) + '.png', Orig_6)


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
            Orig_1 = Original[len(Original) - 15]
            preprocess = prepa[len(Original) - 15]
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
            plt.title('Detected', fontsize=10)
            path1 = "./New_Results/Image_results/Segmented_Image_%s_%s_.png" % (cls[n], i + 1)
            plt.savefig(path1)
            plt.show()
            cv.imwrite('./New_Results/Image_results/Original_image_' + str(i + 1) + '.png', Orig_1)
            cv.imwrite('./New_Results/Image_results/Preprocess_image_' + str(i + 1) + '.png', preprocess)
            cv.imwrite('./New_Results/Image_results/Detected_image_' + str(i + 1) + '.png', seg)


def PLOT_RESULTS():
    plot_Con_results()
    PLot_ROC()
    Plot_Confusion()
    plot_results_Para()
    plot_results_detect()
    Sample_Image_Results()
    Image_Results()


if __name__ == '__main__':
    PLOT_RESULTS()
