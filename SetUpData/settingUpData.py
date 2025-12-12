import matplotlib.pyplot as plt
import fastf1
import numpy as np
import math
import sqlite3
import pandas as pd

from loadSession import getSession
from Functions.calculateCornerFunctions import *
from Functions.identifyDefending import *
from Functions.calculatingLapData import *
from cleaningData import *

database = sqlite3.connect(r"C:\\Users\swavi\Documents\\GitHub\\F1-Stop-Strategy\Databases\database.db")
cursor = database.cursor()


fastf1.set_log_level("ERROR")


def calculatingDriverLaps(year, raceNumber, track):
    session = getSession(year, raceNumber)

    cleaned = cleaningData(session)
    driversLaps = cleaned['driversLaps']
    driversData = cleaned['driversData']
    drivers = cleaned['drivers']
    lapMatrix = cleaned['lapMatrix']

    cursor.execute(
        """INSERT INTO Race (year, circuit)
      VALUES (?, ?)""",
        (year, track)
    )
    raceId = cursor.lastrowid
    database.commit()

    for driver, laps in driversLaps:
        cornerData = setUpCornerData(session)
        currentDriverData = driversData[driversData['Abbreviation'] == driver]

        cursor.execute(
            """INSERT OR IGNORE INTO Driver (code, name, team)
        VALUES (?, ?, ?)""",
            (
                driver,
                str(currentDriverData['FullName'].iloc[0]),
                str(currentDriverData['TeamName'].iloc[0])
            )
        )
        driverId = cursor.lastrowid
        database.commit()

        laps = laps.drop(
            [
                'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
                'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime'
            ],
            axis=1,
            errors='ignore'
        )

        for i in range(laps.shape[0]):
            lapRow = laps.iloc[i]
            lapNumber = lapRow["LapNumber"]

            cursor.execute(
                """INSERT INTO LAP (raceId, driverId, lapNumber, attacking, defending, clean)
          VALUES (?, ?, ?, ?, ?, ?)""",
                (raceId, driverId, lapNumber, False, False, False)
            )
            lapId = cursor.lastrowid
            database.commit()

            lapTelemetry = lapRow.get_telemetry()
            lapTelemetry = lapTelemetry.drop(
                ['Status', 'Z'],
                axis=1,
                errors='ignore'
            )
            lapTelemetry['X'] = lapTelemetry['X'] / 10
            lapTelemetry['Y'] = lapTelemetry['Y'] / 10
            lapTelemetry = lapTelemetry.reset_index(drop=True)
            lapTelemetry = identifyCorner(lapTelemetry, cornerData)
            calculatingData(lapTelemetry, lapId, cursor,session)
            defendingDrivers = getDefendingDrivers(lapTelemetry)

            if len(defendingDrivers) != 0:
                cursor.execute(
                    """UPDATE LAP SET attacking=True WHERE driverId=? AND lapId =?""",
                    (driverId, lapId)
                )
                database.commit()
                lapMatrix.loc[lapNumber, (driver, "defending")] = False
                lapMatrix.loc[lapNumber, (driver, "Drivers Ahead")] = defendingDrivers
                lapMatrix.loc[lapNumber, (driver, lapId)] = lapId
                lapMatrix.loc[lapNumber, (driver, driverId)] = driverId

    identifyIfDefending(lapMatrix, drivers, session, cursor, database)
