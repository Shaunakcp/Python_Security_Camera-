import cv2
import time
import datetime

cap = cv2.VideoCapture(0) # video capture device

face_cascade= cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # set up cascade classifier
body_cascade= cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml") # set up cascade classifier

detection = False 
detection_stopped_time = None 
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v") 

while True:
    _, frame = cap.read() # underscore is placeholder; reading one frame from device and display on screen

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frame to grayscale 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # gray is the image, 1.3 is the scale factor, 5 is the minimum number of neighbors  
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0: # Have we seen any faces or bodies
        if detection: # were we detecting stuff before 
            timer_started = False # are we currently recording 
        else:
            detection = True    
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S") # gets current time
            out = cv2.VideoWriter(
                f"{current_time}.mp4", fourcc, 20.0, frame_size) # makes a new video
            print("Started Recording!")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stop Recording!")
        else: 
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    #for (x, y, width, height) in faces:
    #   cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'): # if you hit 'q' key, program stops
        break 

out.release()
cap.release()
cv2.destroyAllWindows() # destroy the window showing the video capture device 






