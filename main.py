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

lapMatrix = pd.read_pickle("2025Silverstone2.pkl")
lapMatrix = lapMatrix.dropna(subset=[('VER', 'Lap Data')])
# print(lapMatrix.columns)
drivers = lapMatrix.columns.get_level_values(0).unique()
allLaps = []
allDrivers = []
palette = sns.color_palette("hls", len(drivers))
driverColours = {driver: palette[i] for i, driver in enumerate(drivers)}

columns = [
    "throttleStd",
    "throttlePerc100",
    "throttlePerc0",
    "avCornerBrakeDistance",
    "avCornerSpeedDiff",
    "avApexThrottle",
    "throttleOscillation",
    "brakeChanges",
    "coastingPerc",]

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
pca = PCA(n_components=0.95)
Xpca = pca.fit_transform(scaled)
hdb = HDBSCAN(min_cluster_size=6, min_samples=2,
              cluster_selection_method='leaf', metric='euclidean')
hdb.fit_predict(Xpca)
labels = hdb.labels_
df["cluster"] = labels
print(pd.crosstab(df.driver, df.cluster))
cluster_summary = df.groupby("cluster")[columns].mean()
print(cluster_summary)
