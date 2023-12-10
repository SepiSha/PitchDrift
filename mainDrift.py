#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 10:00:13 2021
#########################################################################
# (c) copyright 2021 Sepideh Shafiei (sepideh.shafiee@gmail.com), all rights reserved
#########################################################################
@author: Sepideh Shafiei
"""
import os
import csv
from random import random, choice

import numpy as np
import scipy.signal as scisig
import globs
import readPitchTimeDomain as readPitchTimeDomain
import matplotlib.pyplot as plt
from matplotlib import colors
import extrapolateZeros as extrapolateZeros
import histofinder as histofinder
import findNoteRange as findNoteRange
import sentences as sentences
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

globs.globs()


def main():
    notPAPERprint = False;
    file = 'Example1.mp3'
    outp = "./" + file.replace(".mp3", "_vamp_pyin_pyin_smoothedpitchtrack.csv")
    cmd = './sonic-annotator -t ptfile -w csv %s' % file
    if (os.path.exists(outp)):
        print("No need to do Pyin. File exists", outp)
    else:
        os.system(cmd)
    print("pitch extraction done")
    currentFileNameStr = [outp]
    numFiles = len(currentFileNameStr)

    datamatrix = np.empty(shape=(0, 2), dtype='object')
    with open(currentFileNameStr[0], newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            result = [float(x.strip(' "')) for x in row]
            datamatrix = np.vstack((datamatrix, result))

    globs.X, Y_centOriginal = readPitchTimeDomain.readPitchTimeDomain(datamatrix)
    print("Pitch File read and saved in datamatrix")
    fig, subplot = plt.subplots()
    Y_centNan = np.copy(Y_centOriginal)
    Y_centNan[Y_centNan == 0] = np.nan

    subplot.plot(Y_centNan)
    if (notPAPERprint):
        subplot.set_title("Time-Frequency")
    subplot.set_xlabel('time')
    subplot.set_ylabel('detected frequencies (cents)')
    fig.savefig("Time-Frequency.png")
    fig.show()

    Y_centExtrapolated = extrapolateZeros.extrapolateZeros(Y_centOriginal)
    histogramSmooth, histoStart, histoEnd, Y_centHisto = histofinder.histofinder(Y_centExtrapolated)
    while (Y_centHisto[histoStart] < 3):  # chop off the super-small right and left
        Y_centHisto[histoStart] = 0
        histoStart = histoStart + 1
    while (Y_centHisto[histoEnd] < 3):
        Y_centHisto[histoEnd] = 0
        histoEnd = histoEnd - 1

    fig, subplot = plt.subplots()
    subplot.plot(histogramSmooth)
    subplot.set_xlim(histoStart - 10, histoEnd + 10)
    if (notPAPERprint):
        subplot.set_title("histogram")
    subplot.set_xlabel('detected frequencies (cents)')
    subplot.set_ylabel('number of occurances')
    fig.savefig("mainhisto.png")
    fig.show()

    # looks like the Upper Envelope doesn't do anything, it returns the array itself
    histoUpperEnv = np.real(scisig.hilbert(histogramSmooth))
    trimmedHistoEnv = np.copy(histoUpperEnv)
    trimmedHistoEnv[np.arange(1 - 1, histoStart)] = 0
    trimmedHistoEnv[np.arange(histoEnd, len(trimmedHistoEnv))] = 0

    histoUpperEnv[np.arange(1 - 1, histoStart)] = 0
    UpperEnvmaxlocs, Mdummy = scisig.find_peaks(histoUpperEnv)
    UpperEnvmaxima = histoUpperEnv[UpperEnvmaxlocs]

    histoSmoothmaxlocs, Mdummy = scisig.find_peaks(histogramSmooth, distance=18, prominence=3,
                                                   height=max(max(UpperEnvmaxima / 18), 3))
    histoSmoothmaxima = histogramSmooth[histoSmoothmaxlocs]

    if (len(histoSmoothmaxlocs) == 0):
        raise Exception('no max locs')

    derivative = - histogramSmooth[np.arange(histoStart, histoEnd - 15)] + histogramSmooth[
        np.arange(histoStart + 15, histoEnd)]
    audioShahed, audioShahedSpare, histoSmoothmaxlocs = findNoteRange.findNoteRange(histogramSmooth, 0, derivative,
                                                                                    histoSmoothmaxima,
                                                                                    histoSmoothmaxlocs, histoStart)
    seg = sentences.sentences(Y_centExtrapolated, currentFileNameStr[0])

    LocsTable = np.zeros(shape=(0, 10))
    countTable = 0
    for s in np.arange(1 - 1, len(seg)):
        # print(seg[s].curveBeg)
        histogramSmoothParts, histoStartParts, histoEndParts, Y_centHistoParts = histofinder.histofinder(
            Y_centExtrapolated[seg[s].curveBeg:seg[s].curveEnd])
        histoUpperEnv = np.real(scisig.hilbert(histogramSmoothParts))

        trimmedHistoEnv = np.copy(histoUpperEnv)
        trimmedHistoEnv[np.arange(1 - 1, histoStart)] = 0
        trimmedHistoEnv[np.arange(histoEnd, len(trimmedHistoEnv))] = 0

        histoUpperEnv[np.arange(1 - 1, histoStart)] = 0
        UpperEnvmaxlocs, Mdummy = scisig.find_peaks(histoUpperEnv)
        UpperEnvmaxima = histoUpperEnv[UpperEnvmaxlocs]

        histoSmoothmaxlocsP, Mdummy = scisig.find_peaks(histogramSmoothParts, distance=18, prominence=3,
                                                        height=max(max(UpperEnvmaxima / 18), 3))
        histoSmoothmaxima = histogramSmooth[histoSmoothmaxlocsP]

        if (len(histoSmoothmaxlocs) == 0):
            raise Exception('no max locs')

        if len(histoSmoothmaxlocsP) == 0:
            print('mainDrift******--histoSmoothmaxlocsP is 0')
            # raise Exception('no max locs')
            continue

        derivative = - histogramSmooth[np.arange(histoStart, histoEnd - 15)] + histogramSmooth[
            np.arange(histoStart + 15, histoEnd)]
        audioShahedParts, audioShahedSpareParts, histoSmoothmaxlocsParts = findNoteRange.findNoteRange(
            histogramSmoothParts, s, derivative, histoSmoothmaxima, histoSmoothmaxlocsP, histoStart)
        maxLocsParts = np.zeros((10))
        countTable += 1
        for l in np.arange(0, len(histoSmoothmaxlocsParts)):
            maxLocsParts[l] = histoSmoothmaxlocsParts[l]
        LocsTable = np.append(LocsTable, maxLocsParts.reshape(1, 10), axis=None).reshape(countTable, 10)
        seg[s].hist = histogramSmoothParts
        seg[s].hist /= seg[s].hist.max().item()
        # seg[s].peaks = histoSmoothmaxlocsParts

    x_coordinates = np.zeros(0)
    y_coordinates = np.zeros(0)
    fig, subplot = plt.subplots()
    for m in np.arange(1 - 1, (LocsTable.shape[0])):
        for k in np.arange(1 - 1, 10):
            if (LocsTable[m, k] != 0):
                x_coordinates = np.append(x_coordinates, m)
                y_coordinates = np.append(y_coordinates, int(LocsTable[m, k]) * 10)

    subplot.scatter(x_coordinates, y_coordinates / 10)
    minY = np.nanmin(y_coordinates) / 10
    maxY = np.nanmax(y_coordinates) / 10
    if (notPAPERprint):
        subplot.set_title(" Detected frequencies in segments")
    subplot.set_xlabel('sentence number')
    subplot.set_ylabel('detected frequencies')
    subplot.set_xticks([x for x in range(seg.size)])
    subplot.vlines(subplot.get_xticks(), 0, subplot.get_ylim()[1], linestyles="dashed", linewidth=0.2)
    fig.savefig("frequency-segment.png")
    fig.show()

    matrix = np.zeros((1, 2))
    for f in range(len(x_coordinates)):
        matrix = np.append(matrix, [x_coordinates[f], y_coordinates[f]])
    matrix2 = matrix.reshape(int(len(matrix) / 2), 2)

    clustering = DBSCAN(eps=220, min_samples=2).fit(matrix2)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    samples_w_lbls = np.concatenate((matrix2, labels[:, np.newaxis]), axis=1)
    mm = np.zeros(n_clusters_)
    bb = np.zeros(n_clusters_)
    for p in range(n_clusters_):
        filter = np.asarray([p])
        m3 = samples_w_lbls[np.in1d(samples_w_lbls[:, -1], filter)]
        xsum = np.sum(m3[:, 0])
        x2sum = np.dot(m3[:, 0], m3[:, 0])
        ysum = np.sum(m3[:, 1])
        xysum = np.dot(m3[:, 0], m3[:, 1])
        N = m3.shape[0]
        mm[p] = ((N * xysum) - xsum * ysum) / (N * x2sum - xsum * xsum)
        bb[p] = (ysum - mm[p] * xsum) / N
        print("p, m, b=", p, mm[p], bb[p])

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(matrix2, labels))

    # #############################################################################
    # Plot result of classification

    fig, subplots = plt.subplot_mosaic([["audio", "audio", "audio", "audio", "audio"],
                                        ["cluster", "cluster", "cluster", "cluster", "cluster"],
                                        ["cluster", "cluster", "cluster", "cluster", "cluster"],
                                        ["cluster", "cluster", "cluster", "cluster", "cluster"],
                                        ["cluster", "cluster", "cluster", "cluster", "cluster"]])
    subplots["cluster"].sharex(subplots["audio"])

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors_ = [choice(list(colors.TABLEAU_COLORS.values())) for i in np.linspace(0, 1, len(unique_labels))]
    xy_color = pd.DataFrame()

    for k, col in zip(unique_labels, colors_):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = labels == k

        xy = matrix2[class_member_mask & core_samples_mask]
        subplots["cluster"].plot(xy[:, 0], xy[:, 1] / 10, "o", markerfacecolor=col, markeredgecolor=col,
                                 markersize=14)
        for row_number in range(xy.shape[0]):
            xy_color.loc[tuple(xy[row_number].astype(int))] = col
        xy = matrix2[class_member_mask & ~core_samples_mask]
        subplots["cluster"].plot(xy[:, 0], xy[:, 1] / 10, "o", markerfacecolor=col, markeredgecolor=col,
                                 markersize=6)

    # xy_color.columns /= 10
    # xy_color.fillna((0,0,0), inplace=True)
    # xy_color.replace('nan', (0,0,0), inplace=True)
    for p in range(n_clusters_):
        xrange = range(1 + int(samples_w_lbls.max(0)[0]))
        yrange = mm[p] * xrange + bb[p]
        subplots["cluster"].plot(xrange, yrange / 10)
        subplots["cluster"].set_ylim([minY - 100, maxY + 100])
    if notPAPERprint:
        subplots["cluster"].set_title(" number of clusters: %d" % n_clusters_)

    for idx, segment in np.ndenumerate(seg):
        start_point = idx[0]
        end_point = idx[0] + 1
        num = Y_centExtrapolated[segment.curveBeg: segment.curveEnd].size

        subplots["audio"].plot(np.linspace(start_point, end_point, num),
                               Y_centExtrapolated[segment.curveBeg: segment.curveEnd],
                               linewidth=0.2)

        # color = np.full(fill_value="k", shape=segment.hist.shape)
        # for i in xy_color.columns:
        #     try:
        #         color[int(i)] = xy_color.loc[idx, i]
        #     except IndexError:
        #         pass
        subplots["cluster"].plot((segment.hist[int(minY - 100): int(maxY + 100)] / 1.5) + start_point,
                                    np.arange(subplots["cluster"].get_ylim()[0],
                                              subplots["cluster"].get_ylim()[0] + segment.hist[
                                                                                  int(minY - 100): int(maxY +
                                                                                                       100)
                                                                                  ].size),
                                    linewidth=0.5, color="k")
    subplots["audio"].tick_params(axis="x", which='both', bottom=False, labelbottom=False)
    subplots["audio"].set_ylim(minY, maxY)
    subplots["cluster"].set_xlabel('sentence number')
    subplots["cluster"].set_ylabel('detected frequencies')
    subplots["cluster"].set_xticks([x for x in range(seg.size)])
    # subplots["cluster"].vlines(subplots["cluster"].get_xticks(), 0, subplots["cluster"].get_ylim()[1],
    #                            linestyles="dashed", linewidth=0.2)
    fig.show()
    print('end of prog')


main()
