# import the opencv library
import cv2
import face_detection
import dlib
from scipy.spatial import distance
from imutils import face_utils

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
COUNTER = 0
ALARM_ON = False

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = distance.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio

    print("EAR:", ear)
    
    return ear

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

def draw_points(im, landmarks):
    for landmark in landmarks:
        cv2.circle(im, (landmark.x,landmark.y), radius=2, color=(0, 255, 0), thickness=-1)

# Initialize face detector
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)

# Initialize landmarks detector
predictor = dlib.shape_predictor("predictor.dat")

# define a video capture object
vid = cv2.VideoCapture(0)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Detect faces from the image
    # This will return a tensor with shape [N, 5], 
    # where N is number of faces and the five elements are [xmin, ymin, xmax, ymax, detection_confidence]
    detections = detector.detect(frame)[:, :4]
    draw_faces(frame, detections)
    print("Number of faces detected: {}".format(len(detections)))

    if len(detections) > 0:
        # Detect landmark
        for k, d in enumerate(detections):
            faceBoxRectangle = dlib.rectangle(left=d[0], top=d[1], right=d[2], bottom=d[3])
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, faceBoxRectangle.left(), faceBoxRectangle.top(), faceBoxRectangle.right(), faceBoxRectangle.bottom()))
            
            # Get the landmarks/parts for the face in box d.
            # returns a single full_object_detection
            shape = predictor(frame, faceBoxRectangle)
            draw_points(frame, shape.parts())
            print("Number of landmarks detected: {}".format(shape.num_parts))

            shape = face_utils.shape_to_np(shape)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True
                        # TODO sound alarm
                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()