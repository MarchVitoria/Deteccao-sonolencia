from ear import eye_aspect_ratio, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES
from mar import mouth_aspect_ratio, MOUTH_AR_THRESH, MOUTH_AR_CONSEC_FRAMES
from imutils import face_utils
import cv2
from playsound import playsound

COUNTER = 0
COUNTER_YAWN = 0
ALARM_ON = False

def sound_alarm(path):
	playsound(path, block=False)

def mouth_over_eye(shape, frame):
    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    global COUNTER
    global COUNTER_YAWN
    global ALARM_ON
    shape = face_utils.shape_to_np(shape)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    print("EAR:", ear)

    # extract the mouth coordinates, then use the
    # coordinates to compute the mouth aspect ratio
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

    mouth = shape[mStart:mEnd]
    mar = mouth_aspect_ratio(mouth)

    # check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    if ear < EYE_AR_THRESH:
        COUNTER += 1    

    # otherwise, the mouth aspect ratio is not below the blink
    # threshold, so reset the counter and alarm
    else:
        COUNTER = 0

    # check to see if the eye aspect ratio is below the yawn
    # threshold, and if so, increment the yawn frame counter
    if mar > MOUTH_AR_THRESH:
        COUNTER_YAWN += 1
        
    # otherwise, the mouth aspect ratio is not below the yawn
    # threshold, so reset the counter_yawn and alarm
    else:
        COUNTER_YAWN = 0

    # if the eyes were closed for a sufficient number of
    # then sound the alarm
    if COUNTER >= EYE_AR_CONSEC_FRAMES | COUNTER_YAWN >= MOUTH_AR_CONSEC_FRAMES:
        
        # if the alarm is not on, turn it on
        if not ALARM_ON:
            ALARM_ON = True
            sound_alarm("./data/Sound4.wav")

        # draw an alarm on the frame
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        ALARM_ON = False