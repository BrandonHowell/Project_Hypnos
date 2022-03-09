##############################################
# 
#  Project Hypnos
#
#  Created by: Brandon Howell
#
##############################################

from threading import Thread
from scipy.spatial import distance as dist
import vlc
import cv2
import mediapipe as mp
import keyboard
import time

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def faceMesh():
    cap = cv2.VideoCapture(1)
    frames = 0
    EAR_THRESHOLD = 0.22
    EAR_CONSEC_FRAMES = 35
    keydown = False
    debugToggle = False
    indexToggle = False
    soundPlaying = False
    alertEnabled = True
    alertFile = {'alert':"alerts/default.mp3"}
    occurances = 0
    pTime = 0
    right_eye_indicies = [155,158,160,33,144,153]
    left_eye_indicies = [362,385,387,263,373,380]
    rightEyeCoord = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    leftEyeCoord = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    right_EAR = 0
    left_EAR = 0
    ear = 0
    
    initialize(alertFile)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            
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
                
                ########################
                # DEBUG VISUAL TOGGLES #
                ########################
                if keyboard.is_pressed('d') and keydown == False:
                    if debugToggle == False:
                        debugToggle = True
                    else:
                        debugToggle = False
                    keydown = True
                elif keyboard.is_pressed('i') and keydown == False:
                    if indexToggle == False:
                        indexToggle = True
                    else:
                        indexToggle = False
                    keydown = True
                elif keyboard.is_pressed('d') or keyboard.is_pressed('i'):
                    continue
                else:
                    keydown = False
                
                if debugToggle == True:
                    drawEyeIndicies(image,rightEyeCoord,leftEyeCoord, indexToggle)
                ########################################################################
                # Value Visuals for Demonstration - Will not be visible in final build #
                ########################################################################
                cv2.putText(image, f'left_EAR: {left_EAR}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1) 
                cv2.putText(image, f'right_EAR: {right_EAR}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(image, f'EAR: {EAR_THRESHOLD}|{ear}', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(image, f'Drowsy Frames: {frames}/{EAR_CONSEC_FRAMES}', (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(image, f'Occurences: {occurances}', (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                
                #Detects when the EAR is below the threshold and counts the frames that have passed
                if ear < EAR_THRESHOLD:
                    frames += 1
                    #If the frames exceed the consecutive frame count, we play the sound and add 1 occurance of drowsiness
                    if frames > EAR_CONSEC_FRAMES:
                        #Visual Aid to be Removed in final build
                        cv2.putText(image, f'Drowsiness Detected!', (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        if soundPlaying == False:
                            soundPlaying = True
                            
                            if alertEnabled:
                                occurances += 1
                                #Plays sound on seperate thread to not pause function of the application
                                t = Thread(target=playAlarm, args=(alertFile["alert"],))
                                t.daemon = True
                                t.start()
                            
                else:
                    frames = 0
                    soundPlaying = False
                
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime
            
            cv2.putText(image, f'FPS: {int(fps)}', (10,360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xff == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # eye_data = open("logs/eye_data.txt","wb")
    # output = ""
    # for x in eyes:
    #     eye_data.write("Right: {int(x[0])} | Left: {int(x[1])}\n",)
        
    # eye_data.close()

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

#Personal code for giving a visual representation
def drawEyeIndicies(image,rightEye,leftEye, showIndexs):
    
    index_r = 0
    for rCoord in rightEye:
        cv2.circle(image, (rCoord[0], rCoord[1]), radius=1, color=(255, 255, 0), thickness=2)
        if showIndexs:
            cv2.putText(image, f'{index_r}', (rCoord[0]+5, rCoord[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            index_r += 1
    
    cv2.line(image, (rightEye[0][0],rightEye[0][1]),(rightEye[3][0],rightEye[3][1]), (255,255,255), 1)
    cv2.line(image, (rightEye[1][0],rightEye[1][1]),(rightEye[5][0],rightEye[5][1]), (255,255,255), 1)
    cv2.line(image, (rightEye[2][0],rightEye[2][1]),(rightEye[4][0],rightEye[4][1]), (255,255,255), 1)
    
    index_l = 0
    for lCoord in leftEye:
        cv2.circle(image, (lCoord[0], lCoord[1]), radius=1, color=(255, 255, 0), thickness=2)
        if showIndexs:
            cv2.putText(image, f'{index_l}', (lCoord[0]+5, lCoord[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            index_l += 1
        
    cv2.line(image, (leftEye[0][0],leftEye[0][1]),(leftEye[3][0],leftEye[3][1]), (255,255,255), 1)
    cv2.line(image, (leftEye[1][0],leftEye[1][1]),(leftEye[5][0],leftEye[5][1]), (255,255,255), 1)
    cv2.line(image, (leftEye[2][0],leftEye[2][1]),(leftEye[4][0],leftEye[4][1]), (255,255,255), 1)    
#TEstComment
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