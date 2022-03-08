#Original Template Code and modified for finding eye index points from mediapipe face mesh

import cv2
import mediapipe as mp
import keyboard
import time

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def faceDetect():
    # Face Detect
    cap = cv2.VideoCapture(1)
    with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            
            if not success:
                print("Ignoring empty camera frame.")
                break
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    
            cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xff == 27:
                break

    cap.release()

def faceMesh():
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(1)
    landmark_index = 0
    
    right_eye_upper = 159
    right_eye_lower = 145
    left_eye_upper = 386
    left_eye_lower = 374
    
    right_key_down = False
    left_key_down = False
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7) as face_mesh:
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
                for face_landmarks in results.multi_face_landmarks:
                #     mp_drawing.draw_landmarks(
                #         image=image,
                #         landmark_list = face_landmarks,
                #         connections = mp_face_mesh.FACEMESH_TESSELATION,
                #         landmark_drawing_spec = None,
                #         connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list = face_landmarks,
                        connections = mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style())
                #     mp_drawing.draw_landmarks(
                #         image=image,
                #         landmark_list = face_landmarks,
                #         connections = mp_face_mesh.FACEMESH_IRISES,
                #         landmark_drawing_spec = None,
                #         connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                
                ############################################################# 
                # TEMP CODE FOR SEARCHING CORRECT LANDMARK INDEX FOR EYE LIDS
                #
                if keyboard.is_pressed('right') and right_key_down == False:
                    landmark_index += 1
                    right_key_down = True
                if keyboard.is_pressed('right'):
                    continue
                else:
                    right_key_down = False
                
                if keyboard.is_pressed('left') and left_key_down == False:
                    landmark_index -= 1
                    left_key_down = True
                if keyboard.is_pressed('left'):
                    continue
                else:
                    left_key_down = False
                #############################################################
                
                coord = results.multi_face_landmarks[0].landmark[landmark_index]
                
                x = coord.x
                y = coord.y
                        
                shape = image.shape
                rel_x = int(x * shape[1])
                rel_y = int(y * shape[0])
                
                cv2.circle(image, (rel_x, rel_y), radius=1, color=(25, 99, 180), thickness=2)
                cv2.putText(image, f'Index = {landmark_index}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(image, f'x = {rel_x}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(image, f'x = {rel_y}', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                
            
            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xff == 27:
                break
    cap.release()

def main():
    faceMesh()
    
if __name__ == "__main__":
    main()