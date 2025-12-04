from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
import hdbscan
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()


lapMatrix = pd.read_pickle("2025Silverstone.pkl")
lapMatrix = lapMatrix.dropna(subset=[('VER', 'Lap Data')])
# print(lapMatrix.columns)
drivers = lapMatrix.columns.get_level_values(0).unique()
allLaps = []

columns = [
    "upshifts",
    "downshifts",
    "gearMean",
    "throttleMean",
    "throttleStd",
    "throttlePerc100",
    "throttlePerc0",
    "braking_pct",
    "avCornerBrakeDistance",
    "avCornerSpeedDiff",
    "avApexThrottle",]

for driver in drivers:
    driverData = lapMatrix[(driver, 'Lap Data')]
    for i in range(lapMatrix.shape[0]):
        currentDriverLap = driverData.iloc[i]
        if isinstance(currentDriverLap, dict):
            lapData = list(currentDriverLap.values())
            allLaps.append(lapData)

allLapsDF = pd.DataFrame(allLaps, columns=columns)
scaled = scaler.fit_transform(allLapsDF)

pca = PCA(n_components=2)
Xpca = pca.fit_transform(scaled)
hdb = hdbscan.HDBSCAN(min_cluster_size=20)
hdb.fit(Xpca)
