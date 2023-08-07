from scipy.spatial import distance

MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSEC_FRAMES = 20

def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = distance.euclidean(mouth[1], mouth[7])
    B = distance.euclidean(mouth[2], mouth[6])
    C = distance.euclidean(mouth[3], mouth[5])
    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    D = distance.euclidean(mouth[0], mouth[4])
    # compute the mouth aspect ratio
    mar = (A + B + C) / (2.0 * D)
    # return the mouth aspect ratio

    print("MAR:", mar)
    
    return mar