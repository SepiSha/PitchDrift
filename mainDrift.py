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
label_font_dict = {"size": 100}
legend_font_dict = {"size": 55}
tick_font_dict = {"size": 60}
fig_size_ratio = 3.276
line_width_ratio = 10


def main():
    """
    main routine
    """
    NOT_PAPER_PRINT = False
    EQUAL_SEGMENTS = True
    n_sentences = 23
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
    second_ticks_labels = np.arange(0, int(result[0]), 30)
    globs.X, y_cent_original = readPitchTimeDomain.readPitchTimeDomain(data_matrix)
    print("Pitch File read and saved in datamatrix")
    fig, subplot = plt.subplots(figsize=(20 * fig_size_ratio, 10 * fig_size_ratio))
    y_cent_nan = np.copy(y_cent_original)
    y_cent_nan[y_cent_nan == 0] = np.nan

    subplot.plot(y_cent_nan, linewidth=0.5 * line_width_ratio, color="k", label="frequency")

    if NOT_PAPER_PRINT:
        subplot.set_title("Time-Frequency")
    subplot.set_xticks(np.linspace(0, y_cent_nan.size, second_ticks_labels.size),
                       labels=second_ticks_labels)
    subplot.tick_params(labelsize=tick_font_dict.get("size"))
    subplot.set_xlabel('time (seconds)', fontdict=label_font_dict)
    subplot.set_ylabel('detected frequencies (cents)', fontdict=label_font_dict)
    subplot.legend(fontsize=legend_font_dict.get("size"))
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

    fig, subplot = plt.subplots(figsize=(14 * fig_size_ratio, 10 * fig_size_ratio))
    subplot.plot(histogram_smooth, color="b", label="smoothed histogram", linewidth=line_width_ratio)
    subplot.set_xlim(histo_start - 10, histo_end + 10)
    if NOT_PAPER_PRINT:
        subplot.set_title("histogram")
    subplot.tick_params(labelsize=tick_font_dict.get("size"))
    subplot.set_xlabel('detected frequencies (cents)', fontdict=label_font_dict)
    subplot.set_ylabel('number of occurances', fontdict=label_font_dict)
    subplot.legend(fontsize=legend_font_dict.get("size"))
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
        # seg[s].hist /= seg[s].hist.max().item()
        segment_bounderies.loc[s, "peaks"] = histo_smooth_max_locs_parts
        segment_bounderies.loc[s, "start"] = seg[s].curveBeg
        segment_bounderies.loc[s, "finish"] = seg[s].curveEnd

    x_coordinates_v = np.zeros(0)
    y_coordinates_v = np.zeros(0)
    x_coordinates = np.zeros(0)
    y_coordinates = np.zeros(0)
    fig, subplot = plt.subplots(figsize=(14 * fig_size_ratio, 10 * fig_size_ratio))
    segment_bounderies["tick_positions"] = ((segment_bounderies["finish"] - segment_bounderies["start"]) / 2
                                            ) + segment_bounderies["start"]
    for idx, row in segment_bounderies.iterrows():
        subplot.scatter([(row["tick_positions"]) for i in range(row["peaks"].size)],
                        row["peaks"], color="b", label="main peaks in each sentence" if idx == 0 else "",
                        s=line_width_ratio * 100)
    for m in np.arange(1 - 1, (locs_table.shape[0])):
        for k in np.arange(1 - 1, 10):
            if locs_table[m, k] != 0:
                x_coordinates = np.append(x_coordinates, m)
                y_coordinates = np.append(y_coordinates, int(locs_table[m, k]) * 10)
    for _, row in segment_bounderies.iterrows():
        y_coordinates_v = np.append(y_coordinates_v, row["peaks"] * 10)
        x_coordinates_v = np.append(x_coordinates_v, np.repeat(row["tick_positions"], row["peaks"].size))

    # subplot.scatter(x_coordinates, y_coordinates / 10)
    subplot.set_xticks(segment_bounderies["tick_positions"].values.tolist(),
                       labels=list(segment_bounderies.index + 1),
                       rotation=90,
                       fontdict=tick_font_dict)
    subplot.tick_params(labelsize=tick_font_dict.get("size"))
    min_y = np.nanmin(y_coordinates_v) / 10
    max_y = np.nanmax(y_coordinates_v) / 10
    if NOT_PAPER_PRINT:
        subplot.set_title("Detected frequencies in segments")
    subplot.set_xlabel('sentence number', fontdict=label_font_dict)
    subplot.set_ylabel('detected frequencies', fontdict=label_font_dict)
    subplot.vlines(subplot.get_xticks(), 0, subplot.get_ylim()[1], linestyles="dashed",
                   linewidth=0.2 * line_width_ratio,
                   label="sentence indicator")
    fig.savefig("frequency-segment.png")
    subplot.legend(fontsize=legend_font_dict.get("size"))
    fig.show()

    matrix = np.zeros((1, 2))
    matrix_v = np.zeros((1, 2))
    for f in range(len(x_coordinates)):
        matrix = np.append(matrix, [x_coordinates[f], y_coordinates[f]])
    for f in range(len(x_coordinates_v)):
        matrix_v = np.append(matrix_v, [x_coordinates_v[f], y_coordinates_v[f]])
    matrix_2 = matrix.reshape(int(len(matrix) / 2), 2)
    matrix_v = matrix_v.reshape(int(len(matrix_v) / 2), 2)

    clustering = DBSCAN(eps=300, min_samples=3).fit(matrix_2)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    samples_w_labels = np.concatenate((matrix_v, labels[:, np.newaxis]), axis=1)
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
    # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(matrix_2, labels))

    # #############################################################################
    # Plot result of classification

    fig, subplots = plt.subplot_mosaic([["audio", "audio", "audio", "audio"],
                                        ["cluster", "cluster", "cluster", "cluster"],
                                        ["cluster", "cluster", "cluster", "cluster"],
                                        ["cluster", "cluster", "cluster", "cluster"],
                                        ["cluster", "cluster", "cluster", "cluster"]],
                                       figsize=(20 * fig_size_ratio, 10 * fig_size_ratio))
    subplots["cluster"].sharex(subplots["audio"])

    unique_labels = set(labels)
    colors_ = [choice(list(colors.TABLEAU_COLORS.values())) for i in np.linspace(0, 1, len(unique_labels))]
    xy_color = pd.DataFrame(columns=segment_bounderies["tick_positions"].astype(int),
                            index=np.concatenate([i.flatten() for i in segment_bounderies["peaks"]]))
    label_colors = {}
    label_legned = {-1: "noise",
                    0: "first cluster",
                    1: "second cluster"}
    for k, col in zip(unique_labels, colors_):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        label_colors[k] = col
        xy = matrix_v[class_member_mask]
        subplots["cluster"].plot(xy[:, 0], xy[:, 1] / 10, "o", markerfacecolor=col, markeredgecolor=col,
                                 markersize=3 * line_width_ratio,
                                 label=label_legned[k] if k == 0 or k == 1 or k == -1 else str(k + 1) + "th cluster",
                                 linewidth=line_width_ratio)
        for row_number in range(xy.shape[0]):
            if isinstance(col, str):
                xy_color.loc[int(xy[row_number][1] / 10), int(xy[row_number][0])] = col

    # xy_color.columns /= 10
    xy_color.replace('nan', np.nan, inplace=True)
    xy_color.replace(np.nan, )
    xy_color = xy_color.transpose()
    for p in range(n_clusters_):
        xrange = range(1 + int(samples_w_labels.max(0)[0]))
        yrange = mm[p] * xrange + bb[p]
        subplots["cluster"].plot(xrange, yrange / 10, linestyle="dashed", color="k", linewidth=0.6 * line_width_ratio,
                                 label="clusters indicator lines" if p == 0 else "")
        subplots["cluster"].set_ylim([min_y - 100, max_y + 100])
    if NOT_PAPER_PRINT:
        subplots["cluster"].set_title(" number of clusters: %d" % n_clusters_)
    maximum_hist = np.nanmax(np.concatenate([s.hist for s in seg]))
    for idx, segment in np.ndenumerate(seg):
        start_point = segment.curveBeg
        end_point = segment.curveEnd
        audio_line = y_cent_extrapolated[segment.curveBeg: segment.curveEnd]
        audio_line = np.where(audio_line == 0, np.nan, audio_line)
        num = audio_line.size
        subplots["audio"].plot(np.linspace(start_point, end_point, num),
                               audio_line,
                               linewidth=0.2 * line_width_ratio,
                               color="b" if idx[0] % 2 == 1 else "r")

        scatter_x = (((segment.hist[int(min_y - 100): int(max_y + 100)]) / maximum_hist) * (
                seg[-1].curveEnd / len(seg))) + start_point
        scatter_y = np.arange(subplots["cluster"].get_ylim()[0],
                              subplots["cluster"].get_ylim()[0] + segment.hist[
                                                                  int(min_y - 100): int(max_y +
                                                                                        100)
                                                                  ].size)
        colors_ = [colors.to_rgb("#000000") for i in range(scatter_x.size)]

        for color_y, color in xy_color.iloc[idx[0]].items():
            if isinstance(color, str) and color != 'nan':
                colors_[int(color_y - min_y + 80): int(color_y - min_y + 120)] = [colors.to_rgb(color) for i in
                                                                                  range(40)]

        subplots["cluster"].scatter(scatter_x, scatter_y, s=0.005 * line_width_ratio, c=colors_)

    subplots["audio"].tick_params(axis="x", which='both', bottom=False, labelbottom=False,
                                  labelsize=tick_font_dict.get("size"))
    subplots["audio"].set_ylim(min_y - 500, max_y + 500)
    subplots["cluster"].set_xlabel('time (seconds)', fontdict=label_font_dict)
    subplots["cluster"].set_ylabel('detected frequencies', fontdict=label_font_dict)
    subplots["cluster"].set_xticks(np.linspace(0, seg[-1].curveEnd, second_ticks_labels.size),
                                   labels=second_ticks_labels)
    subplots["audio"].tick_params(labelsize=tick_font_dict.get("size"))
    subplots["cluster"].tick_params(labelsize=tick_font_dict.get("size"))
    subplots["audio"].legend()
    subplots["cluster"].legend()
    cluster_legend = subplots["cluster"].get_legend()
    audio_legend = subplots["audio"].get_legend()
    legend_handles = cluster_legend.legend_handles.copy()
    legend_handles.extend(audio_legend.legend_handles)
    legend_labels = [handle.get_label() for handle in legend_handles]
    subplots["cluster"].legend(handles=legend_handles, labels=legend_labels, loc="lower right",
                             fontsize=legend_font_dict.get("size"))
    subplots["audio"].legend().remove()
    fig.show()
    fig.savefig("classification")
    print('end of prog')


main()
