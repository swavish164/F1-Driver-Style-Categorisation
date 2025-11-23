import fastf1
import os
import pickle


temp_file = "session.pkl"
session = None


def loadSession():
    global session
    if os.path.exists(temp_file):
        with open(temp_file, 'rb') as f:
            session = pickle.load(f)
    else:
        session = fastf1.get_session(2025, 'Silverstone', 'R')
        session.load(laps=True, messages=True)
        with open(temp_file, "wb") as f:
            pickle.dump(session, f)


loadSession()
