import loadSession
import pandas as pd
import fastf1

fastf1.set_log_level("ERROR")

session = loadSession.session

laps = session.laps
sessionStatus = session.track_status

fullLaps = laps[
    (laps['Sector1Time'] > pd.Timedelta(0)) &
    (laps['Sector2Time'] > pd.Timedelta(0)) &
    (laps['Sector3Time'] > pd.Timedelta(0)) &
    (laps['Deleted'] == False)
].copy()

deletedLaps = laps[laps['Deleted'] == True]

fullLaps['SC'] = False
fullLaps['VSC'] = False
fullLaps['Yellow Flag'] = False

fullLaps['TrackStatus'] = fullLaps['TrackStatus'].fillna('')

sc_mask = fullLaps['TrackStatus'].str.contains('4', regex=False)
vsc_mask = fullLaps['TrackStatus'].str.contains('6', regex=False)
yellow_mask = fullLaps['TrackStatus'].str.contains('2', regex=False)
# red_mask = fullLaps['TrackStatus'].str.contains('5', regex=False)

fullLaps.loc[sc_mask, 'SC'] = True
fullLaps.loc[vsc_mask, 'VSC'] = True
fullLaps.loc[yellow_mask, 'Yellow Flag'] = True
# fullLaps.loc[vsc_mask, 'Red Flag'] = True

fullGreenLaps = fullLaps[(fullLaps['SC'] == False) &
                         (fullLaps['VSC'] == False) &
                         (fullLaps['Yellow Flag'] == False)
                         ]
driversLaps = fullGreenLaps.groupby("Driver")

drivers = fullLaps["Driver"].unique()
metrics = ["attacking", "defending", "Drivers Ahead", "Lap Data"]
columns = pd.MultiIndex.from_product([drivers, metrics])
lapNumbers = sorted(fullLaps["LapNumber"].unique())
lapMatrix = pd.DataFrame(index=lapNumbers, columns=columns)
lapMatrix = lapMatrix.astype(object)

lapRow = fullLaps.iloc[0]
lapTelemetry = lapRow.get_telemetry()

for driver, laps in driversLaps:
    for i in range(laps.shape[0]):
        lapRow = laps.iloc[i]
        lapNumber = lapRow["LapNumber"]
        lapTelemetry = lapRow.get_telemetry()
        minDistanceToDriverAhead = lapTelemetry["DistanceToDriverAhead"].min()
        driversAhead = set(lapTelemetry["DriverAhead"].dropna().unique())
        attacking = minDistanceToDriverAhead < 1
        defendingDriver = lapTelemetry["DriverAhead"] if attacking else 0

        lapMatrix.loc[lapNumber, (driver, "attacking")] = attacking
        lapMatrix.loc[lapNumber, (driver, "defending")] = False
        lapMatrix.loc[lapNumber, (driver, "Drivers Ahead")] = driversAhead
        lapMatrix.at[lapNumber, (driver, "Lap Data")] = lapTelemetry

for lapNumber in lapMatrix.index:
    for defender in drivers:
        defending = False
        for attacker in drivers:
            lapData = lapMatrix.loc[lapNumber, (attacker, "Lap Data")]
            if lapData is None:
                continue
            driversAhead = lapMatrix.loc[lapNumber,
                                         (attacker, "Drivers Ahead")]
            if not pd.isna(driversAhead) and defender in driversAhead:
                defending = True
                break
        lapMatrix.loc[lapNumber, (defender, "defending")] = defending

lapMatrix.to_pickle("lapMatrix.pkl")
