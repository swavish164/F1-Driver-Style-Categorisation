import fastf1
import os
import pickle

session = None
DataDir = "sessions"
os.makedirs(DataDir, exist_ok=True)


def getSession(year, raceNumber):
    global session
    fileName = str(year) + "RaceNumber" + str(raceNumber)

    if os.path.exists(fileName):
        with open(fileName, 'rb') as f:
            session = pickle.load(f)
        return session

    session = fastf1.get_session(year, raceNumber, 'R')
    session.load(laps=True, telemetry=True, messages=True)

    with open(fileName, "wb") as f:
        pickle.dump(session, f)

    return session
