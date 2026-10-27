# Dataset instructions

Download the **Metro Interstate Traffic Volume** dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

Place the downloaded CSV (or CSV.GZ file) in this folder. The training script looks for:
- `Metro_Interstate_Traffic_Volume.csv`
- `Metro_Interstate_Traffic_Volume.csv.gz`
- `Metro_Interstate_Traffic_Volume.csv.zip`

Source: Hogue, J. (2019), UCI Machine Learning Repository. DOI: 10.24432/C5X60B.
License reported by UCI: Creative Commons Attribution 4.0 International (CC BY 4.0).

The dataset contains hourly westbound I-94 traffic volume and weather/holiday features. It is not a direct, independently measured congestion-label dataset. In this project, Low/Moderate/High classes are derived from traffic-volume quantiles calculated on the training period only. Do not describe these derived labels as verified ground-truth congestion.
