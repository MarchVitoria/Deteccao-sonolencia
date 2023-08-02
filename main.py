# import the opencv library
import cv2
import face_detection

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

# Initialize detector
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    #Detectar face
    # Detect faces from the image
    # Note:
    #   1. The input image must be a byte array of dimension HxWx3.
    #   2. The return value is a Nx5 (for S3FD) or a Nx15 (for RetinaFace) matrix,
    #      in which N is the number of detected faces. The first 4 columns store 
    #      (in this order) the left, top, right, and bottom coordinates of the 
    #      detected face boxes. The 5th columns stores the detection confidences.
    #      The remaining columns store the coordinates (in the order of x1, y1, x2,
    #      y2, ...) of the detected landmarks.
    detections = detector.detect(frame)[:, :4]
    draw_faces(frame, detections)

    #Se existir face
    #   Detectar landmark
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()