import cv2
import numpy as np
import math

def empty(a):
    pass
def endpoint_finder(start_point, slope_path, distance, bounce):
    if type(slope_path) == float:
        angle = np.rad2deg(np.arctan(slope_path))
    elif slope_path[0][0] - slope_path[1][0] != 0:
        if slope_path[0][0] > slope_path[1][0]:
            neg = 180
        else:
            neg = 0
        slope_path = (slope_path[0][1] - slope_path[1][1]) / (slope_path[0][0] - slope_path[1][0])

        angle = np.rad2deg(np.arctan(slope_path)) + neg
    elif slope_path[0][1] < slope_path[1][1]:
        angle = 90
    else:
        angle = 270
    if bounce:
        if (start_point[0] == radiusLine[0][0]) | (start_point[0] == radiusLine[1][0]):
            angle = -angle + 180
        else:
            angle = -angle
    x_path = np.int(start_point[0] + distance * np.cos(np.deg2rad(angle)))
    y_path = np.int(start_point[1] + distance * np.sin(np.deg2rad(angle)))
    return x_path, y_path
def slope_intersect(start_point, slope_path):
    if slope_path[0][0] - slope_path[1][0] != 0:
        sloped = (slope_path[0][1] - slope_path[1][1]) / (slope_path[0][0] - slope_path[1][0])
        b = start_point[1] - sloped * start_point[0]
        sloped = round(sloped, 2)
        b = round(b, 2)

    else:
        sloped = None
        b = None
    return sloped, b
def collision_point_finder(endpoint, start_point, slope_path):
    collisions = None, None  # Clear all collisions every frame
    if (endpoint[0] >= radiusLine[0][0]) & (endpoint[0] <= radiusLine[1][0]) & (endpoint[1] >= radiusLine[0][1]) & (endpoint[1] <= radiusLine[1][1]):
        # print("PATH LINE DOESNT HIT WALL")                                        # Prediction line has no collisions
        pass
    elif not ((endpoint[0] > radiusLine[0][0]) & (endpoint[0] < radiusLine[1][0]) | (endpoint[1] > radiusLine[0][1]) & (
            endpoint[1] < radiusLine[1][1])):
        # print("PATH LINE IN CORNER")                                              # Prediction line has collisions with corners
        slope_b = slope_intersect(start_point, slope_path)
        if slope_b[0] is not None:
            if slope_b[0] != 0:
                if (radiusLine[0][0] >= endpoint[0]) & (radiusLine[0][1] >= endpoint[1]):  # Left top corner
                    # print("Left top corner")
                    # print(slope_b[0],radiusLine[0][0],slope_b[1])
                    y = int(slope_b[0] * radiusLine[0][0] + slope_b[1])
                    x = int((radiusLine[0][1] - slope_b[1]) / slope_b[0])
                    if y < radiusLine[0][1]:
                        collisions = (x, radiusLine[0][1])
                    else:
                        collisions = (radiusLine[0][0], y)
                elif (radiusLine[1][0] <= endpoint[0]) & (radiusLine[0][1] >= endpoint[1]):  # Right top corner
                    # print("Right top corner")
                    y = int(slope_b[0] * radiusLine[1][0] + slope_b[1])
                    x = int((radiusLine[0][1] - slope_b[1]) / slope_b[0])
                    if y < radiusLine[0][1]:
                        collisions = (x, radiusLine[0][1])
                    else:
                        collisions = (radiusLine[1][0], y)
                elif (radiusLine[0][0] >= endpoint[0]) & (radiusLine[1][1] <= endpoint[1]):  # Left bot corner
                    # print("Left bot corner")
                    y = int(slope_b[0] * radiusLine[0][0] + slope_b[1])
                    x = int((radiusLine[1][1] - slope_b[1]) / slope_b[0])
                    if y > radiusLine[1][1]:
                        collisions = (x, radiusLine[1][1])
                    else:
                        collisions = (radiusLine[0][0], y)
                elif (radiusLine[1][0] <= endpoint[0]) & (radiusLine[1][1] <= endpoint[1]):  # Right bot corner
                    # print("Right bot corner")
                    y = int(slope_b[0] * radiusLine[1][0] + slope_b[1])
                    x = int((radiusLine[1][1] - slope_b[1]) / slope_b[0])
                    if y > radiusLine[1][1]:
                        collisions = (x, radiusLine[1][1])
                    else:
                        collisions = (radiusLine[1][0], y)
                else:
                    print("??? point: ", endpoint)
                    print("wall chords: ", radiusLine)

    else:
        # print("PATH LINE IN BORDER")                       # Prediction line has collisions with borders
        slope_b = slope_intersect(start_point, slope_path)
        if slope_b[0] == 0:
            # print("Line is horizontal")
            if radiusLine[0][0] > endpoint[0]:  # Left wall
                collisions = (radiusLine[0][0], endpoint[1])
            else:  # Right wall
                collisions = (radiusLine[1][0], endpoint[1])
        elif slope_b[0] is None:
            # print("Line is vertical")
            if radiusLine[0][1] > endpoint[1]:  # Top wall
                collisions = (endpoint[0], radiusLine[0][1])
            else:  # Bot wall
                collisions = (endpoint[0], radiusLine[1][1])
        else:
            # print("Line isn't perpendicular")
            if radiusLine[0][0] > endpoint[0]:  # Left wall
                # print("Left wall")
                y = int(slope_b[0] * radiusLine[0][0] + slope_b[1])
                collisions = (radiusLine[0][0], y)
            elif radiusLine[1][0] < endpoint[0]:  # Right wall
                # print("Right wall")
                y = int(slope_b[0] * radiusLine[1][0] + slope_b[1])
                collisions = (radiusLine[1][0], y)
            elif radiusLine[0][1] > endpoint[1]:  # Top wall
                # print("Top wall")
                x = int((radiusLine[0][1] - slope_b[1]) / slope_b[0])
                collisions = (x, radiusLine[0][1])
            elif radiusLine[1][1] < endpoint[1]:  # Bot wall
                # print("Bot wall")
                x = int((radiusLine[1][1] - slope_b[1]) / slope_b[0])
                collisions = (x, radiusLine[1][1])
    return collisions
def show_lines(line_cords):
    for o, line in enumerate(line_cords):
        # print("lines length: ", len(line_cords))
        if o + 1 == len(line_cords):
            # print("OWO")
            cv2.line(blank, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 1)  # Predict path
            cv2.circle(blank, (line[1][0], line[1][1]), 20, (99, 99, 99), 1)  # failed predicts
        else:
            cv2.line(blank, (line[0][0], line[0][1]), (line[2][0], line[2][1]), (0, 0, 255), 1)  # good path
            cv2.circle(blank, (line[2][0], line[2][1]), 4, (0, 0, 255), -1)  # good predicts
            cv2.line(blank, (line[1][0], line[1][1]), (line[2][0], line[2][1]), (99, 99, 99), 1)  # failed path
            cv2.circle(blank, (line[1][0], line[1][1]), 4, (99, 99, 99), -1)  # failed predicts
def all_collisions(bounce, all_bounce, slopes, end_point):
    j = 0
    while bounce != (None, None):

        if j > 3:  # keep from crashing
            j = 0
            print("Bounce Max")
            break
        start_point = bounce
        length = ((bounce[0] - end_point[0]) ** 2 + (bounce[1] - end_point[1]) ** 2) ** .5
        end_point = endpoint_finder(bounce, slopes, length, True)
        slopes = (bounce[0], bounce[1]), (end_point[0], end_point[1])
        bounce = collision_point_finder(end_point, start_point, slopes)
        # print("\nall_bounce", all_bounce)
        # print("\nstart_point", start_point)
        # print("\nend_point", end_point)
        # print("\nbounce", bounce)

        all_bounce = np.append(all_bounce, [[start_point, end_point, bounce]], axis=0)
        # finalEndPoint = pointEnd
        j += 1
    return all_bounce

global radiusLine

radius = 20  # Radius of ball
table = (800, 425)  # Table size
ballCircle = (200, 237)  # Ball location
purBall = (540,300)
bounds = 100  # Buffer area around table
wallLine = [(bounds, bounds), (table[0], table[1])]  # Visual bounds
radiusLine = [(bounds + radius, bounds + radius), (table[0] - radius, table[1] - radius)]  # Invisible bounds

cv2.namedWindow("Slide Bar")  # Make window for trackbars
cv2.resizeWindow("Slide Bar", 500, 250)
cv2.createTrackbar("Cue X1", "Slide Bar", 230, 1000, empty)
cv2.createTrackbar("Cue Y1", "Slide Bar", 300, 1000, empty)
cv2.createTrackbar("Cue X2", "Slide Bar", 380, 1000, empty)
cv2.createTrackbar("Cue Y2", "Slide Bar", 330, 1000, empty)
cv2.createTrackbar("Range", "Slide Bar", 250, 1000, empty)
i = 0
while True:

    cueX1 = cv2.getTrackbarPos("Cue X1", "Slide Bar")
    cueY1 = cv2.getTrackbarPos("Cue Y1", "Slide Bar")
    cueX2 = cv2.getTrackbarPos("Cue X2", "Slide Bar")
    cueY2 = cv2.getTrackbarPos("Cue Y2", "Slide Bar")
    D = cv2.getTrackbarPos("Range", "Slide Bar")

    blank = np.zeros((wallLine[1][1] + bounds, wallLine[1][0] + bounds, 3), np.uint8)  # Create canvas the size of table
    cueStick = [(cueX1, cueY1), (cueX2, cueY2)]  # Location of cue stick

    pointEnd = endpoint_finder(ballCircle, cueStick, D, False)
    collision = collision_point_finder(pointEnd, ballCircle, cueStick)  # Finds collisions
    endPoint = pointEnd
    # print("cueBallList", ballCircle)
    lines = np.array([[ballCircle, endPoint, collision]], dtype="object")
    # print("post", lines)
    slopePoints = ballCircle, collision
    lines = all_collisions(collision, lines, slopePoints, endPoint)


    cv2.rectangle(blank, wallLine[0], wallLine[1], (0, 255, 0), 2)  # Wall
    cv2.rectangle(blank, radiusLine[0], radiusLine[1], (99, 99, 99), 1, 1)  # Radius wall
    cv2.circle(blank, ballCircle, radius, (255, 255, 255), -1)  # Ball
    cv2.circle(blank, ballCircle, 4, (0, 0, 255), 0)  # Ball dot
    cv2.circle(blank, purBall, radius, (255, 0, 255), -1)  # Purple Ball
    cv2.circle(blank, purBall, 3, (0, 0, 0), 0)  # Ball dot
    cv2.arrowedLine(blank, cueStick[0], cueStick[1], (255, 255, 255), 3)  # Cue

    show_lines(lines)

    cv2.imshow("kk", blank)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break