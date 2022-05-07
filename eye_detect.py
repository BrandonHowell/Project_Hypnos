##############################################
# 
#  Project Hypnos
#
#  Created by: Brandon Howell
#
##############################################

from datetime import datetime as dt
from math import ceil
from threading import Thread
import numpy as np
from scipy.spatial import distance as dist
import vlc
import cv2
import mediapipe as mp
import time
import sqlite3
import PySimpleGUI as sg 

# GUI Layout
sg.theme('Dark Blue 2')

col_webcam_layout = [
    [sg.Text("Camera Preview", size=(60,1), justification="center")],
    [sg.Image(filename="", key="cam1")],
    [sg.Button('Preview', size=(10,1), font='Helvetica 14')]
]
col_webcam = sg.Column(col_webcam_layout, element_justification='center')

menu_layout = [
    [sg.Button('Start', size=(10,1), font='Helvetica 14')],
    [sg.Button('Stop', size=(10,1), font='Helvetica 14')],
    [sg.HorizontalSeparator()],
    [sg.Button('Calendar', size=(10,1), font='Helvetica 14')],
    [sg.HorizontalSeparator()],
    [sg.Button('Exit', size=(10,1), font='Helvetica 14')]
]

col_menu = sg.Column(menu_layout, element_justification='center')

main_layout = [col_webcam, col_menu]

rowfooter = [sg.Image(filename="", key="-IMAGEBOTTOM-")]
layout = [main_layout, rowfooter]

window = sg.Window("Project Hypnos", layout, no_titlebar=False, alpha_channel=1, grab_anywhere=False, return_keyboard_events=True, location=(100, 100))


def guiController(event): 
    
    return

def faceMesh():
    
    #Mediapipe Face Detection Library
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    #Main Face Detection Variables
    cap = cv2.VideoCapture(1)
    frame_count = 0
    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 25
    soundPlaying = False
    alertEnabled = True
    alertFile = {'alert':"alerts/default.mp3"}
    right_eye_indicies = [155,158,160,33,144,153]
    left_eye_indicies = [362,385,387,263,373,380]
    rightEyeCoord = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    leftEyeCoord = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    right_EAR = 0
    left_EAR = 0
    ear = 0

    # Sqlite Connection and Data
    con = sqlite3.connect('data/hypnos_data.db')
    userid = 1
    occurances = 0
    datetime_of_occurance = dt.now()  
    
    # GUI Camera Settings
    camera_Width  = 320 # 480 # 640 # 1024 # 1280
    camera_Heigth = 240 # 320 # 480 # 780  # 960
    frameSize = (camera_Width, camera_Heigth)
    time.sleep(2.0)
    recording = False
    preview = False
    img = np.full((camera_Heigth,camera_Width), 0)
    
    initialize(alertFile)
    databaseInitialize(con)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4) as face_mesh:
        while True:
            success, image = cap.read()
            event, values = window.read(timeout=1)
            
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            elif event == 'Start':
                recording = True
            elif event == 'Preview' and preview == False:
                preview = True
            elif event == 'Preview' and preview == True:
                preview = False
                imgbytes = cv2.imencode(".png", img)[1].tobytes()
                window["cam1"].update(data=imgbytes)
            elif event == 'Stop':
                recording = False
                preview = False
                imgbytes = cv2.imencode(".png", img)[1].tobytes()
                window["cam1"].update(data=imgbytes)
            
            if preview == True:
                frame = cv2.resize(image, frameSize)
                imgbytes = cv2.imencode(".png", frame)[1].tobytes()
                window["cam1"].update(data=imgbytes)
            
            if not success:
                print("Ignoring empty camera frame.")
                break
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                
                leftEyeCoord = getEyeCoordinates(image, results, left_eye_indicies)
                rightEyeCoord = getEyeCoordinates(image, results, right_eye_indicies)
                
                left_EAR = eye_aspect_ratio(leftEyeCoord)
                right_EAR = eye_aspect_ratio(rightEyeCoord)
                
                ear = (left_EAR + right_EAR) / 2.0
                
                #Detects when the EAR is below the threshold and counts the frames that have passed
                if ear < EAR_THRESHOLD and recording == True:
                    frame_count += 1
                    #If the frames exceed the consecutive frame count, we play the sound and add 1 occurance of drowsiness
                    if frame_count > EAR_CONSEC_FRAMES:
                        #Visual Aid to be Removed in final build
                        if soundPlaying == False:
                            soundPlaying = True
                            
                            if alertEnabled:
                                occurances += 1
                                #Plays sound on seperate thread to not pause function of the application
                                t = Thread(target=playAlarm, args=(alertFile["alert"],))
                                t.daemon = True
                                t.start()
                                datetime_of_occurance = dt.now()
                                dbPush(userid, con, datetime_of_occurance, occurances)
                            
                else:
                    frame_count = 0
                    soundPlaying = False
            
            #cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xff == 27:
                break
    
    window.close()
    con.close()
    cap.release()
    cv2.destroyAllWindows()

def databaseInitialize(con):
    cur = con.cursor()
    
    cur.execute("CREATE TABLE if not exists User(uid INTEGER PRIMARY KEY AUTOINCREMENT, FirstName name, LastName name)")
    cur.execute("CREATE TABLE if not exists Month(MonthID INTEGER  PRIMARY KEY AUTOINCREMENT, uid int, month int)")
    cur.execute("CREATE TABLE if not exists Week(WeekID INTEGER  PRIMARY KEY AUTOINCREMENT, MonthID int, week int)")
    cur.execute("CREATE TABLE if not exists Day(DayID INTEGER  PRIMARY KEY AUTOINCREMENT, WeekID int, day int, occurance_total int)")
    cur.execute("CREATE TABLE if not exists Occurance(id INTEGER PRIMARY KEY AUTOINCREMENT, DayID int, time_of_occurance time)")
    
    con.commit()

def dbPush(uid, con, dto, occ):
    cur = con.cursor()
    
    dto = dt.now()
    month = dto.month
    week = week_of_month(dto)
    day = dto.day
    time = dto.strftime("%H:%M:%S")
    
    cur.execute("INSERT INTO Month(uid, month) SELECT ?, ? WHERE NOT EXISTS (SELECT * FROM MONTH WHERE uid = ? AND month = ?);",(uid, month, uid, month))
    cur.execute("SELECT MonthID FROM Month WHERE uid = ? and month = ?",(uid, month))
    monthID = cur.fetchone()
    cur.execute("INSERT INTO Week(MonthID, week) SELECT ?, ? WHERE NOT EXISTS (SELECT * FROM Week WHERE MonthID = ? AND week = ?);",(monthID[0], week, monthID[0], week))
    cur.execute("SELECT WeekID FROM Week WHERE MonthID = ? and week = ?",(monthID[0], week))
    weekID = cur.fetchone()
    cur.execute("INSERT INTO Day(WeekID, day) SELECT ?, ? WHERE NOT EXISTS (SELECT * FROM Day WHERE WeekID = ? AND day = ?);",(weekID[0], day, weekID[0], day))
    cur.execute("SELECT DayID FROM Day WHERE WeekID = ? and day = ?",(weekID[0], day))
    dayID = cur.fetchone()
    cur.execute("INSERT INTO Occurance(DayID, time_of_occurance) SELECT ?, ? WHERE NOT EXISTS (SELECT * FROM Occurance WHERE DayID = ? AND time_of_occurance = ?);",(dayID[0], time, dayID[0], time))
    cur.execute("SELECT COUNT(DayID) FROM Occurance WHERE DayID = ?;",(dayID[0],))
    occurance_total = cur.fetchone()
    cur.execute("UPDATE Day SET occurance_total = ? WHERE DayID = ?;",(occurance_total[0], dayID[0]))
    
    con.commit()
    
def week_of_month(dto):
    first_day = dto.replace(day=1)
    date_of_month = dto.day
    adjusted_dom = date_of_month + first_day.weekday()
    return int(ceil(adjusted_dom/7.0))

#Initializes audio player to avoid start-up lag of first occurance of drowsiness
def initialize(alertFile):
    t = Thread(target=playAlarm, args=(alertFile["alert"],))
    t.daemon = True
    t.start()

#Function to be ran on seperate thread
def playAlarm(alert):
    p = vlc.MediaPlayer(alert)
    p.play()
    time.sleep(3)
    p.stop()   

#Retrieves and converts eye index locations to (x,y) coordinates on screen
def getEyeCoordinates(image, results, eye):
    eye_coords = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    
    for index in range(6):
        coord = results.multi_face_landmarks[0].landmark[eye[index]]
    
        x = coord.x
        y = coord.y
            
        shape = image.shape
        rel_x = int(x * shape[1])
        rel_y = int(y * shape[0])
        
        eye_coords[index][0] = rel_x
        eye_coords[index][1] = rel_y
    
    return eye_coords

#Re-used Code from Adrian Rosebrock (pyimagesearch.com)
#Calculates the Eye Aspect Ratio (EAR) based on 6 landmark points around the eye
def eye_aspect_ratio(eye):
    # Vertical Landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Horizontal Landmarks
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    # Return EAR
    return ear

def main():
    faceMesh()
    
if __name__ == "__main__":
    main()