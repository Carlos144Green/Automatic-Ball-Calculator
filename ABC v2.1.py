import cv2
import numpy as np
import sys


def empty(a):  # this is to pass useless params
    pass
# Border Detection
def outer_censorship(empty_table):
    hsv = cv2.cvtColor(empty_table, cv2.COLOR_BGR2HSV)                              # Convert to HSV
    ranges = cv2.inRange(hsv, np.array([0, 105, 209]), np.array([11, 255, 255]))    # Get mask for color
    color = cv2.Canny(ranges, 100, 100)                                             # Get tape borders
    kernel = np.ones((20, 20))                                                      # Set brush size
    color_dilate = cv2.dilate(color, kernel, iterations=1)                          # Dilate to close gap of tape
    color_dilate = cv2.medianBlur(color_dilate, 15, 1)                              # Blur to further close gap
    contours, rank = cv2.findContours(color_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # Get table edges
    return contours
def getIntersection(line1, line2):
    s1 = np.array(line1[0])                     # 3rd party code
    e1 = np.array(line1[1])                     # Idk how it works

    s2 = np.array(line2[0])                     # it just gets 2 lines and outputs the intersections
    e2 = np.array(line2[1])

    a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
    b1 = s1[1] - (a1 * s1[0])

    a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
    b2 = s2[1] - (a2 * s2[0])

    if abs(a1 - a2) < sys.float_info.epsilon:
        return False

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return (x, y)
def inner_censorship(empty_table):
    hsv = cv2.cvtColor(empty_table, cv2.COLOR_BGR2HSV)                          # Convert to HSV
    ranges = cv2.inRange(hsv, np.array([0, 0, 250]), np.array([180, 90, 255]))  # Get mask for color
    canny = cv2.Canny(ranges, 950, 475)                                         # Get all borders
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 120)                          # Find the lines
    hori = []                                                                   # All horizontal lines
    verti = []                                                                  # All vertical lines
    for line in lines:                                                          # Find details for every line
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if abs(x1 - x2) <= 50:                                                  # If tips of line have similar x cords
            if len(hori) < 4:                                                   # and we have less than 4 hori lines
                hori.append(((x1, y1), (x2, y2)))                               # Mark it as horizontal line
        elif abs(y1 - y2) <= 25:                                                # Same here but for vertical
            if len(verti) < 4:
                verti.append(((x1, y1), (x2, y2)))

    if len(hori) == 4 & len(verti) == 4:
        ba_points = []
        fr_points = []
        all_points = []
        fake_points = []

        hori.sort()
        verti.sort()
        num_loops = 4
        for i in range(num_loops):
            xy = getIntersection(hori[i], verti[i])
            if not np.isnan(xy[0]):
                xy = [round(num) for num in xy]
                ba_points.append(xy)

        verti.sort(reverse=True)
        for i in range(num_loops):
            xy = getIntersection(hori[i], verti[i])
            if not np.isnan(xy[0]):
                xy = [round(num) for num in xy]
                fr_points.append(xy)

        if len(ba_points) == 4:
            ba_points.pop(0)
            ba_points.pop(len(ba_points) - 1)
            all_points.append(ba_points[0])
            all_points.append(ba_points[1])
            if len(fr_points) == 4:
                fr_points.pop(0)
                fr_points.pop(len(fr_points) - 1)
                all_points.append(fr_points[0])
                all_points.append(fr_points[1])
        if len(all_points) == 4:
            if all_points[0][1] > all_points[3][1]:
                y_1 = all_points[0][1]
            else:
                y_1 = all_points[3][1]
            if all_points[0][0] > all_points[2][0]:
                x_1 = all_points[0][0]
            else:
                x_1 = all_points[2][0]
            if all_points[1][1] < all_points[2][1]:
                y_2 = all_points[1][1]
            else:
                y_2 = all_points[2][1]
            if all_points[1][0] < all_points[3][0]:
                x_2 = all_points[1][0]
            else:
                x_2 = all_points[3][0]
            # Tweak Section
            x_1 += 5
            y_1 += 5
            x_2 -= 10
            y_2 -= 5

    return (x_1, y_1), (x_2, y_2)
def full_censorship(empty_table, result_type):
    canvas = np.zeros((empty_table.shape[0], empty_table.shape[1], 3), np.uint8)
    contours = outer_censorship(empty_table)
    in_points = inner_censorship(empty_table)
    if result_type == "balls":
        canvas.fill(255)
    elif result_type == "radiusLine":
        return in_points
    cv2.fillPoly(canvas, pts=contours, color=(255, 255, 255))                        # Show table edges
    # print(in_points[0])
    # print(in_points[1])
    # print("owo")
    cv2.rectangle(canvas,   (in_points[0]), (in_points[1]), (0, 0, 0), -1)
    canvas = cv2.bitwise_not(canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    return canvas
# Ball Detection
def remove_cue(balls, cue):  # This finds the cue ball from the list of balls and locates it
    tweak = 8
    for j in range(np.size(balls, 0)):
        if ((balls[j][0] - tweak < cue[0][0]) & (balls[j][0] + tweak > cue[0][0]) & (
                balls[j][1] - tweak < cue[0][1]) & (balls[j][1] + tweak > cue[0][1])):
            return j
def pre_processing(stream):
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
def cue_ball_finder(cue_ball, border_bound, radius):
    cue_ball = cv2.bitwise_and(cue_ball, border_bound)  # Keep all overlapping white sections
    cue_ball_list = cv2.HoughCircles(cue_ball, cv2.HOUGH_GRADIENT, 1, radius, param1=11, param2=10, minRadius=8,
                                     maxRadius=12)  # Find cue ball
    if cue_ball_list is not None:
        cue_ball_list = np.round(cue_ball_list[0, :]).astype("int")  # Make cue ball x&y int
        if np.size(cue_ball_list, 0) >= 2:  # Make sure just one cue ball is found
            # print("MULTIPLE CUE FOUND: ", np.size(cue_ball_list, 0))
            for j in range(np.size(cue_ball_list, 0) - 1):  # Delete all other cue balls
                # print("Cue Array: ", cue_ball_list)
                cue_ball_list = np.delete(cue_ball_list, 1, 0)
    return cue_ball_list
def ball_finder(stream, border_bounds, cue_ball_list):
    grey = stream.copy()
    grey = cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY)  # Make gray
    color_table = cv2.bitwise_and(grey, border_bounds)  # Keep all overlapping white sections
    th2 = cv2.adaptiveThreshold(color_table, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
    circles = cv2.HoughCircles(th2, cv2.HOUGH_GRADIENT, 1, rad, param1=11, param2=10, minRadius=8, maxRadius=12)
    if circles is not None:
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
def endpoint_finder(start_point, slope_path, distance, bounce):
    if type(slope_path) == float:
        # print("float slope")
        angle = np.rad2deg(np.arctan(slope_path))
    elif slope_path[0][0] - slope_path[1][0] != 0:
        if slope_path[0][0] > slope_path[1][0]:
            neg = 180
        else:
            neg = 0
        slope_path = (slope_path[0][1] - slope_path[1][1]) / (slope_path[0][0] - slope_path[1][0])
        # print(np.arctan(slope_path))

        angle = np.rad2deg(np.arctan(slope_path)) + neg
    elif slope_path[0][1] < slope_path[1][1]:
        angle = 90
    else:
        angle = 270
    # print("deg:", angle)
    if bounce:
        if (start_point[0] == radiusLine[0][0]) | (start_point[0] == radiusLine[1][0]):
            angle = -angle + 180
        else:
            angle = -angle
    x_path = np.int(start_point[0] + distance * np.cos(np.deg2rad(angle)))
    y_path = np.int(start_point[1] + distance * np.sin(np.deg2rad(angle)))
    return x_path, y_path
def slope_intersect(start_point, slope_path):
    # print("start_point, slope_path",start_point, slope_path)
    if slope_path[0][0] - slope_path[1][0] != 0:
        sloped = (slope_path[0][1] - slope_path[1][1]) / (slope_path[0][0] - slope_path[1][0])
        b = start_point[1] - sloped * start_point[0]
        sloped = round(sloped, 2)
        b = round(b, 2)

        # print("slope int equ: Y=", slope, "X+", b)
    else:
        sloped = None
        b = None
        # print("X=", ball_circle[0])
    return sloped, b
def collision_point_finder(endpoint, start_point, slope_path):
    collisions = None, None  # Clear all collisions every frame
    if (endpoint[0] >= radiusLine[0][0]) & (endpoint[0] <= radiusLine[1][0]) & (endpoint[1] >= radiusLine[0][1]) & (
            endpoint[1] <= radiusLine[1][1]):
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
        print("lines: ", line[2])
        # if line[2][0] is not None:
        #     print("line[0]",line)
        #     np.rint(line)
        if o + 1 == len(line_cords):
            # print("OWO")
            cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 0, 255), 1)  # Predict path
            cv2.circle(img, (line[1][0], line[1][1]), 20, (99, 99, 99), 1)  # failed predicts
        else:
            cv2.line(img, (line[0][0], line[0][1]), (line[2][0], line[2][1]), (0, 0, 255), 1)  # good path
            cv2.circle(img, (line[2][0], line[2][1]), 4, (0, 0, 255), -1)  # good predicts
            cv2.line(img, (line[1][0], line[1][1]), (line[2][0], line[2][1]), (99, 99, 99), 1)  # failed path
            cv2.circle(img, (line[1][0], line[1][1]), 4, (99, 99, 99), -1)  # failed predicts
def all_collisions(bounce, all_bounce, slopes, end_point):
    j = 0
    while bounce != (None, None):

        if j > 3:  # keep from crashing
            j = 0
            print("Max Bounce")
            break
        start_point = bounce
        if bounce[0] is None:
            print("Oh shit")
        if bounce[1] is None:
            print("Oh shit")
        distance = ((bounce[0] - end_point[0]) ** 2 + (bounce[1] - end_point[1]) ** 2) ** .5
        end_point = endpoint_finder(bounce, slopes, distance, True)
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
global past_balls

cap = cv2.VideoCapture("/Users/lolc4/PycharmProjects/Test1/White1.mp4")
success, img = cap.read()  # Starts up first frame
img = cv2.resize(img, None, fx=.6, fy=.6, interpolation=cv2.INTER_CUBIC)  # Resize

tableCensor = full_censorship(img, "table")
ballsCensor = full_censorship(img, "balls")

cueStick = None
D = 600
radius = 11
tableBounds1, tableBounds2 = full_censorship(img, "radiusLine")
radiusLine = ((tableBounds1[0]+radius,tableBounds1[1]+radius),(tableBounds2[0]-radius, tableBounds2[1]-radius))

while True:
    success, img = cap.read()
    timer = cv2.getTickCount()  # Counts the time from here to the end for fps
    img = cv2.resize(img, None, fx=.6, fy=.6, interpolation=cv2.INTER_CUBIC)  # Resize
    raw = img.copy()
    rad = 10
    thicc = 28

    cueBall = pre_processing(img)
    cueBallList = cue_ball_finder(cueBall, ballsCensor, rad)
    ball_finder(img, ballsCensor, cueBallList)
    gIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(raw, 50, 150)
    censoredStream = cv2.bitwise_and(canny, tableCensor)

    cueLines = cv2.HoughLinesP(censoredStream, 1, np.pi / 180, threshold=100, minLineLength=20, maxLineGap=80)
    if cueLines is not None:
        # print(lines[0])
        # for line in lines:
        for x1, y1, x2, y2 in cueLines[0]:
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5
            if length <= 1000:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if cueBallList is not None:
                    # print("cue ball", cueBallList[0])
                    length1 = ((x1 - cueBallList[0][0]) ** 2 + (y1 - cueBallList[0][1]) ** 2) ** .5
                    length2 = ((x2 - cueBallList[0][0]) ** 2 + (y2 - cueBallList[0][1]) ** 2) ** .5
                    if length1 <= length2:
                        cv2.circle(img, (x1, y1), 3, (200, 100, 100), 2)
                        cueStick = [(x2, y2), (x1, y1)]
                        c2bDistance = length1
                    else:
                        cv2.circle(img, (x2, y2), 3, (200, 100, 100), 3)
                        cueStick = [(x1, y1), (x2, y2)]
                        c2bDistance = length2
                    if c2bDistance <= 100:
                        cv2.putText(img, "TAKE AIM!", (75, 100), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255), 2)
                        cv2.circle(img, (cueBallList[0][0], cueBallList[0][1]), 75, (0, 0, 255), 1)
                        if cueStick is not None:
                            if cueBallList is not None:
                                pointEnd = endpoint_finder(cueBallList[0], cueStick, D, False)
                                collision = collision_point_finder(pointEnd, cueBallList[0],
                                                                   cueStick)  # Finds collisions
                                endPoint = pointEnd
                                # print("cueBallList",cueBallList[0])
                                lines = np.array([[(cueBallList[0][0], cueBallList[0][1]), endPoint, collision]],
                                                 dtype="int")
                                # print("post", lines)
                                slopePoints = cueBallList[0], collision
                                lines = all_collisions(collision, lines, slopePoints, endPoint)  ###
                                show_lines(lines)



    cv2.rectangle(img, radiusLine[0], radiusLine[1], (0, 200, 0), 1)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)  # FPS reader
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_COMPLEX, .7, (100, 0, 255), 2)
    cv2.imshow("Table1", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
