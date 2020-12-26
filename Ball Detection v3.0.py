import cv2
import numpy as np

def empty(a):       # this is to pass useless params
    pass
def pre_processing_cue(stream):
    kernel = np.ones([3, 3], np.uint8)
    cue_ball = stream.copy()
    cue_ball = cv2.cvtColor(cue_ball, cv2.COLOR_BGR2HSV)  # Make HSV
    lower = np.array([0, 0, 170], dtype=np.uint8)  # Lower color
    upper = np.array([255, 50, 255], dtype=np.uint8)  # Upper color
    cue_ball = cv2.medianBlur(cue_ball, 3)
    cue_ball = cv2.inRange(cue_ball, lower, upper)  # Find all color within range
    cue_ball = cv2.erode(cue_ball, kernel)
    cue_ball = cv2.morphologyEx(cue_ball, cv2.MORPH_OPEN, kernel)  # Close false open
    return cue_ball
def border_censorship(stream, bbox):
    blank = np.zeros((stream.shape[0], stream.shape[1], 3), np.uint8)  # Make black canvas
    border_bound = cv2.rectangle(blank, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 255, 255), -1)  # Mask canvas
    border_bound = cv2.cvtColor(border_bound, cv2.COLOR_BGR2GRAY)  # Make gray to bitwise
    return border_bound
def cue_ball_finder(cue_ball, border_bound, radius):
    cue_ball = cv2.bitwise_and(cue_ball, border_bound)  # Keep all overlapping white sections
    cue_ball_list = cv2.HoughCircles(cue_ball, cv2.HOUGH_GRADIENT, 1, radius, param1=11, param2=10, minRadius=8, maxRadius=12)  # Find cue ball
    if cue_ball_list is not None:
        cue_ball_list = np.round(cue_ball_list[0, :]).astype("int")  # Make cue ball x&y int
        if np.size(cue_ball_list, 0) >= 2:  # Make sure just one cue ball is found
            print("MULTIPLE CUE FOUND: ", np.size(cue_ball_list, 0))
            for j in range(np.size(cue_ball_list, 0) - 1):  # Delete all other cue balls
                print("Cue Array: ", cue_ball_list)
                cue_ball_list = np.delete(cue_ball_list, 1, 0)
    return cue_ball_list
def remove_cue(balls, cue):             # This finds the cue ball from the list of balls and locates it
    tweak = 7
    for j in range(np.size(balls, 0)):
        if ((balls[j][0] - tweak < cue[0][0]) & (balls[j][0] + tweak > cue[0][0]) & (
                balls[j][1] - tweak < cue[0][1]) & (balls[j][1] + tweak > cue[0][1])):
            return j
def ball_finder(stream, border_bounds, cue_ball_list, past_circles):
    grey = stream.copy()
    grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)  # Make gray
    color_table = cv2.bitwise_and(grey, border_bounds)  # Keep all overlapping white sections
    th2 = cv2.adaptiveThreshold(color_table, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,10)
    circles = cv2.HoughCircles(th2, cv2.HOUGH_GRADIENT, 1, rad, param1=11, param2=10, minRadius=8, maxRadius=12)
    if circles is not None:
        circles = matcher(circles, past_circles)
        circles = np.round(circles[0, :]).astype("int")  # Make all balls x&y int
        if cue_ball_list is not None:  # Checks if a cue ball is found
            cv2.putText(img, "Found!", (75, 75), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            cv2.circle(img, (cue_ball_list[0][0], cue_ball_list[0][1]), 10, (0, 255, 0), 3)

            cue_index = remove_cue(circles, cue_ball_list)  # Removes the cue ball from the balls list
            # print("cueIndex: ", cue_index)
            if cue_index is not None:  # Checks if a cue ball is found in the cue list
                circles = np.delete(circles, cue_index, 0)  # Mark cue ball with diff color
        else:
            cv2.putText(img, "Lost!", (75, 75), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255), 2)
        for i in range(np.size(circles, 0)):  # Display all other balls
            cv2.circle(img, (circles[i][0], circles[i][1]), 10, (255, 255, 0), 3)
# def matcher(pool_balls, past_balls):
    new_balls=[[]]
    if past_balls is not None:
        for i in range(np.size(pool_balls, 0)):  # Display all other balls
            for j in range(np.size(past_balls, 0)):  # Display all other balls
                shift = ((past_balls[j][0]-pool_balls[i][0])**2+(past_balls[j][1]-pool_balls[i][1])**2)**.5
                # if shift <5:
                #     new

    # elif True:
    #     pass
    else:
        past_balls = pool_balls

    print(past_balls)

    return pool_balls

global past_circles
past_circles = None


cap = cv2.VideoCapture("White.mp4")
success, img = cap.read()                                                   # Starts up first frame
img = cv2.resize(img, None, fx=.6, fy=.6, interpolation=cv2.INTER_CUBIC)    # Resize
bBox = cv2.selectROI("Tracking", img, True)                                 # Region of Interest
cv2.destroyWindow("Tracking")                                               # Close window out

while True:
    success, img = cap.read()
    timer = cv2.getTickCount()                                           # Counts the time from here to the end for fps
    img = cv2.resize(img, None, fx=.6, fy=.6, interpolation=cv2.INTER_CUBIC)    # Resize
    rad = 10

    cueBall = pre_processing_cue(img)
    border = border_censorship(img, bBox)
    cueBallList = cue_ball_finder(cueBall, border, rad)
    ball_finder(img, border, cueBallList, past_circles)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)             # FPS reader
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_COMPLEX, .7, (100, 0, 255), 2)
    cv2.imshow("Table1", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break