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
from random import choice
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
from sklearn.cluster import DBSCAN
from sklearn import metrics

globs.globs()


def main():
    """
    main routine
    """
    NOT_PAPER_PRINT = False
    EQUAL_SEGMENTS = False
    n_sentences = 24
    file = 'Example1.mp3'
    out_p = "./" + file.replace(".mp3", "_vamp_pyin_pyin_smoothedpitchtrack.csv")
    cmd = './sonic-annotator -t ptfile -w csv %s' % file
    if os.path.exists(out_p):
        print("No need to do Pyin. File exists", out_p)
    else:
        os.system(cmd)
    print("pitch extraction done")
    current_file_name_str = [out_p]
    data_matrix = np.empty(shape=(0, 2), dtype='object')
    with open(current_file_name_str[0], newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            result = [float(x.strip(' "')) for x in row]
            data_matrix = np.vstack((data_matrix, result))

    globs.X, y_cent_original = readPitchTimeDomain.readPitchTimeDomain(data_matrix)
    print("Pitch File read and saved in datamatrix")
    fig, subplot = plt.subplots(figsize=(20, 10))
    y_cent_nan = np.copy(y_cent_original)
    y_cent_nan[y_cent_nan == 0] = np.nan

    subplot.plot(y_cent_nan, linewidth=0.5, color="k")

    if NOT_PAPER_PRINT:
        subplot.set_title("Time-Frequency")
    subplot.set_xlabel('time')
    subplot.set_ylabel('detected frequencies (cents)')
    fig.savefig("Time-Frequency.png")
    fig.show()

    y_cent_extrapolated = extrapolateZeros.extrapolateZeros(y_cent_original)
    histogram_smooth, histo_start, histo_end, y_cent_histo = histofinder.histofinder(y_cent_extrapolated)
    while y_cent_histo[histo_start] < 3:  # chop off the super-small right and left
        y_cent_histo[histo_start] = 0
        histo_start += 1
    while y_cent_histo[histo_end] < 3:
        y_cent_histo[histo_end] = 0
        histo_end -= 1

    fig, subplot = plt.subplots()
    subplot.plot(histogram_smooth)
    subplot.set_xlim(histo_start - 10, histo_end + 10)
    if NOT_PAPER_PRINT:
        subplot.set_title("histogram")
    subplot.set_xlabel('detected frequencies (cents)')
    subplot.set_ylabel('number of occurances')
    fig.savefig("mainhisto.png")
    fig.show()

    # looks like the Upper Envelope doesn't do anything, it returns the array itself
    histo_upper_env = np.real(scisig.hilbert(histogram_smooth))
    trimmed_histo_env = np.copy(histo_upper_env)
    trimmed_histo_env[np.arange(1 - 1, histo_start)] = 0
    trimmed_histo_env[np.arange(histo_end, len(trimmed_histo_env))] = 0

    histo_upper_env[np.arange(1 - 1, histo_start)] = 0
    upper_env_max_locs, m_dummy = scisig.find_peaks(histo_upper_env)
    upper_env_maxima = histo_upper_env[upper_env_max_locs]

    histo_smooth_max_locs, m_dummy = scisig.find_peaks(histogram_smooth, distance=18, prominence=3,
                                                       height=max(max(upper_env_maxima / 18), 3))
    histo_smooth_maxima = histogram_smooth[histo_smooth_max_locs]

    if len(histo_smooth_max_locs) == 0:
        raise Exception('no max locs')

    derivative = - histogram_smooth[np.arange(histo_start, histo_end - 15)] + histogram_smooth[
        np.arange(histo_start + 15, histo_end)]
    audio_shahed, audio_shahed_spare, histo_smooth_max_locs = findNoteRange.findNoteRange(histogram_smooth, 0,
                                                                                          derivative,
                                                                                          histo_smooth_maxima,
                                                                                          histo_smooth_max_locs,
                                                                                          histo_start)
    seg = sentences.sentences(y_cent_extrapolated, current_file_name_str[0], EQUAL_SEGMENTS, n_sentences)

    locs_table = np.zeros(shape=(0, 10))
    count_table = 0
    segment_bounderies = pd.DataFrame(columns=["start", "finish", "peaks"], index=range(seg.size))
    for s in np.arange(1 - 1, len(seg)):
        histogram_smooth_parts, histo_start_parts, histo_end_parts, y_cent_histo_parts = histofinder.histofinder(
            y_cent_extrapolated[seg[s].curveBeg:seg[s].curveEnd])
        histo_upper_env = np.real(scisig.hilbert(histogram_smooth_parts))

        trimmed_histo_env = np.copy(histo_upper_env)
        trimmed_histo_env[np.arange(1 - 1, histo_start)] = 0
        trimmed_histo_env[np.arange(histo_end, len(trimmed_histo_env))] = 0

        histo_upper_env[np.arange(1 - 1, histo_start)] = 0
        upper_env_max_locs, m_dummy = scisig.find_peaks(histo_upper_env)
        upper_env_maxima = histo_upper_env[upper_env_max_locs]

        histo_smooth_max_locs_p, m_dummy = scisig.find_peaks(histogram_smooth_parts, distance=18, prominence=3,
                                                             height=max(max(upper_env_maxima / 18), 3))
        histo_smooth_maxima = histogram_smooth[histo_smooth_max_locs_p]

        if len(histo_smooth_max_locs) == 0:
            raise Exception('no max locs')

        if len(histo_smooth_max_locs_p) == 0:
            print('main drift******--histo_smooth_max_locs_p is 0')
            continue

        derivative = - histogram_smooth[np.arange(histo_start,
                                                  histo_end - 15)] + histogram_smooth[np.arange(histo_start + 15,
                                                                                                histo_end)]
        audio_shahed_parts, audio_shahed_spare_parts, histo_smooth_max_locs_parts = findNoteRange.findNoteRange(
            histogram_smooth_parts, s, derivative, histo_smooth_maxima, histo_smooth_max_locs_p, histo_start)
        max_locs_parts = np.zeros(10)
        count_table += 1
        for li in np.arange(0, len(histo_smooth_max_locs_parts)):
            max_locs_parts[li] = histo_smooth_max_locs_parts[li]
        locs_table = np.append(locs_table, max_locs_parts.reshape(1, 10)).reshape(count_table, 10)
        seg[s].hist = histogram_smooth_parts
        seg[s].hist /= seg[s].hist.max().item()
        segment_bounderies.loc[s, "peaks"] = histo_smooth_max_locs_parts
        segment_bounderies.loc[s, "start"] = seg[s].curveBeg
        segment_bounderies.loc[s, "finish"] = seg[s].curveEnd

    x_coordinates = np.zeros(0)
    y_coordinates = np.zeros(0)
    fig, subplot = plt.subplots()
    segment_bounderies["tick_positions"] = ((segment_bounderies["finish"] - segment_bounderies["start"]) / 2
                                            ) + segment_bounderies["start"]
    for idx, row in segment_bounderies.iterrows():
        subplot.scatter([(row["tick_positions"]) for i in range(row["peaks"].size)],
                        row["peaks"], color="b")
    for m in np.arange(1 - 1, (locs_table.shape[0])):
        for k in np.arange(1 - 1, 10):
            if locs_table[m, k] != 0:
                x_coordinates = np.append(x_coordinates, m)
                y_coordinates = np.append(y_coordinates, int(locs_table[m, k]) * 10)

    # subplot.scatter(x_coordinates, y_coordinates / 10)
    subplot.set_xticks(segment_bounderies["tick_positions"].values.tolist(),
                       labels= segment_bounderies.index.tolist())
    min_y = np.nanmin(y_coordinates) / 10
    max_y = np.nanmax(y_coordinates) / 10
    if NOT_PAPER_PRINT:
        subplot.set_title("Detected frequencies in segments")
    subplot.set_xlabel('sentence number')
    subplot.set_ylabel('detected frequencies')
    # subplot.set_xticks([x for x in range(seg.size)])
    subplot.vlines(subplot.get_xticks(), 0, subplot.get_ylim()[1], linestyles="dashed", linewidth=0.2)
    fig.savefig("frequency-segment.png")
    fig.show()

    matrix = np.zeros((1, 2))
    for f in range(len(x_coordinates)):
        matrix = np.append(matrix, [x_coordinates[f], y_coordinates[f]])
    matrix_2 = matrix.reshape(int(len(matrix) / 2), 2)

    clustering = DBSCAN(eps=250, min_samples=3).fit(matrix_2)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    samples_w_labels = np.concatenate((matrix_2, labels[:, np.newaxis]), axis=1)
    mm = np.zeros(n_clusters_)
    bb = np.zeros(n_clusters_)
    for p in range(n_clusters_):
        filter_ = np.asarray([p])
        m3 = samples_w_labels[np.in1d(samples_w_labels[:, -1], filter_)]
        x_sum = np.sum(m3[:, 0])
        x_2_sum = np.dot(m3[:, 0], m3[:, 0])
        y_sum = np.sum(m3[:, 1])
        x_y_sum = np.dot(m3[:, 0], m3[:, 1])
        n = m3.shape[0]
        mm[p] = ((n * x_y_sum) - x_sum * y_sum) / (n * x_2_sum - x_sum * x_sum)
        bb[p] = (y_sum - mm[p] * x_sum) / n
        print("p, m, b=", p, mm[p], bb[p])

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(matrix_2, labels))

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

        xy = matrix_2[class_member_mask & core_samples_mask]
        subplots["cluster"].plot(xy[:, 0], xy[:, 1] / 10, "o", markerfacecolor=col, markeredgecolor=col,
                                 markersize=4)
        for row_number in range(xy.shape[0]):
            xy_color.loc[tuple(xy[row_number].astype(int))] = col
        xy = matrix_2[class_member_mask & ~core_samples_mask]
        subplots["cluster"].plot(xy[:, 0], xy[:, 1] / 10, "o", markerfacecolor=col, markeredgecolor=col,
                                 markersize=4)

    xy_color.columns /= 10
    xy_color.replace('nan', np.nan, inplace=True)
    for p in range(n_clusters_):
        xrange = range(1 + int(samples_w_labels.max(0)[0]))
        yrange = mm[p] * xrange + bb[p]
        subplots["cluster"].plot(xrange, yrange / 10)
        subplots["cluster"].set_ylim([min_y - 100, max_y + 100])
    if NOT_PAPER_PRINT:
        subplots["cluster"].set_title(" number of clusters: %d" % n_clusters_)

    for idx, segment in np.ndenumerate(seg):
        start_point = idx[0]
        end_point = idx[0] + 1
        num = y_cent_extrapolated[segment.curveBeg: segment.curveEnd].size
        subplots["audio"].plot(np.linspace(start_point, end_point, num),
                               y_cent_extrapolated[segment.curveBeg: segment.curveEnd],
                               linewidth=0.2,
                               color="b" if idx[0] % 2 == 1 else "r")

        scatter_x = (segment.hist[int(min_y - 100): int(max_y + 100)] / 1.5) + start_point
        scatter_y = np.arange(subplots["cluster"].get_ylim()[0],
                              subplots["cluster"].get_ylim()[0] + segment.hist[
                                                                  int(min_y - 100): int(max_y +
                                                                                        100)
                                                                  ].size)
        colors_ = [colors.to_rgb("#000000") for i in range(scatter_x.size)]
        try:
            for color_y, color in xy_color.loc[idx].items():
                if isinstance(color, str) and color != 'nan':
                    colors_[int(color_y - min_y + 80): int(color_y - min_y + 120)] = [colors.to_rgb(color) for i in
                                                                                      range(40)]

        except KeyError:
            pass
        subplots["cluster"].scatter(scatter_x, scatter_y, s=0.01, c=colors_)

    subplots["audio"].tick_params(axis="x", which='both', bottom=False, labelbottom=False)
    subplots["audio"].set_ylim(min_y - 500, max_y + 500)
    subplots["cluster"].set_xlabel('sentence number')
    subplots["cluster"].set_ylabel('detected frequencies')
    subplots["cluster"].set_xticks([x for x in range(seg.size)])
    fig.show()
    fig.savefig("classification")
    print('end of prog')


main()
