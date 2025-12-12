import math
import numpy as np
import pandas as pd


def identifyIfDefending(lapMatrix, drivers, session, cursor, database):
    for lapNumber in lapMatrix.index:
        for defender in drivers:
            for attacker in drivers:
                lapData = lapMatrix.loc[lapNumber, (attacker, "Lap Data")]

                if lapData is None:
                    continue

                driversAheadData = lapMatrix.loc[lapNumber, (attacker, "Drivers Ahead")]
                driverNumber = session.get_driver(defender)['DriverNumber']
                

                if driversAheadData is None or (isinstance(driversAheadData, float) and math.isnan(driversAheadData)):
                    driversAhead = []
                elif isinstance(driversAheadData, list):
                    driversAhead = driversAheadData
                elif isinstance(driversAheadData, (np.ndarray, pd.Series)):
                    driversAhead = list(driversAheadData)

                if driverNumber in driversAhead:
                    driverId = lapMatrix.loc[lapNumber, (attacker,"driverId")]
                    lapId = lapMatrix.loc[lapNumber, (attacker,"lapId")]

                    cursor.execute(
                        """UPDATE LAP SET defending=? WHERE driverId=? AND raceId =?""",
                        (1,driverId, lapId)
                    )
                    database.commit()


def getDefendingDrivers(lapTelemetry):
    defendingDrivers = set(
        (lapTelemetry[lapTelemetry['DistanceToDriverAhead'] < 1])[
            "DriverAhead"]
        .dropna()
        .unique()
    )
    defending = [item for item in defendingDrivers]
    return defending
