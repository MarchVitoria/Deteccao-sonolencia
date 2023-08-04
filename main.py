# import the opencv library
import cv2
import face_detection
import dlib

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

# Initialize face detector
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)

# Initialize landmarks detector
predictor = dlib.shape_predictor("predictor.dat")

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
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
            shape = predictor(frame, faceBoxRectangle)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                    shape.part(1)))
            # TODO Draw the face landmarks on the screen.
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()