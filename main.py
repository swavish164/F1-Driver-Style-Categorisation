from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg")
plt.ioff()
scaler = StandardScaler()

lapMatrix = pd.read_pickle("2025Silverstone.pkl")
lapMatrix = lapMatrix.dropna(subset=[('VER', 'Lap Data')])
# print(lapMatrix.columns)
drivers = lapMatrix.columns.get_level_values(0).unique()
allLaps = []
allDrivers = []
palette = sns.color_palette("hls", len(drivers))
driverColours = {driver: palette[i] for i, driver in enumerate(drivers)}

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

data = {col: [] for col in columns}


for driver in drivers:
    driverData = lapMatrix[(driver, 'Lap Data')]
    for i in range(lapMatrix.shape[0]):
        currentDriverLap = driverData.iloc[i]
        if isinstance(currentDriverLap, dict):
            lapData = list(currentDriverLap.values())
            allLaps.append(lapData)
            allDrivers.append(driver)
            for i, col in enumerate(columns):
                value = lapData[i]
                data[col].append({"driver": driver, "value": value})

df = pd.DataFrame({
    "driver": allDrivers,
    **{col: [row["value"] for row in data[col]] for col in columns}
})

sns.pairplot(
    df,
    vars=columns,
    hue="driver",
    palette=driverColours,
    plot_kws={"s": 10, "alpha": 0.6}
)
plt.show()

allLapsDF = pd.DataFrame(allLaps, columns=columns)
scaled = scaler.fit_transform(allLapsDF)

pca = PCA(n_components=2)
Xpca = pca.fit_transform(scaled)
hdb = HDBSCAN(min_cluster_size=20, max_cluster_size=50)
hdb.fit(Xpca)
labels = hdb.labels_
