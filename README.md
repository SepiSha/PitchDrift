# PitchDrift
# (c) copyright 2022 Sepideh Shafiei (sepideh.shafiee@gmail.com), all rights reserved
CITATIONS: If you are using PitchDrift in research work for publication please cite: 

We use DBSCAN clustering and linear regression to find the intonation/pitch drift during the course of a solo singing (our data is monophonic voice)

- Input: mp3 file
- Output: Pitch Drift graph
- Steps: 
        1. Pitch recognition of the voice (Dependencies: Sonice Annotator: https://vamp-plugins.org/sonic-annotator/)
        2. Segmentation of the performance and finding the musical phrases
        3. Finding the pitch histogram of each phrase
        4. Finding the peaks of the histograms and use the peaks as frequency data points for clustering
        5. Using DBSCAN for clustering the frequency data points in each segment
        6. Use linear regression to find the slope of the line in each cluster to see the pattern of the intonation drift
