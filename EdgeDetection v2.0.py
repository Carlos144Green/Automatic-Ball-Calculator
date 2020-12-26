import cv2
import numpy as np
import sys

def empty(a):
   pass
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
    s1 = np.array(line1[0])
    e1 = np.array(line1[1])

    s2 = np.array(line2[0])
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
    hsv = cv2.cvtColor(empty_table, cv2.COLOR_BGR2HSV)  # Convert to HSV
    ranges = cv2.inRange(hsv, np.array([0, 0, 250]), np.array([180, 90, 255]))  # Get mask for color
    canny = cv2.Canny(ranges, 950, 475)
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 120)
    hori = []
    verti = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if abs(x1 - x2) <= 50:
            if len(hori) < 4:
                hori.append(((x1, y1), (x2, y2)))
        elif abs(y1 - y2) <= 25:
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
    cv2.fillPoly(canvas, pts=contours, color=(255, 255, 255))                        # Show table edges
    cv2.rectangle(canvas,(in_points[0]), (in_points[1]), (0, 0, 0), -1)
    canvas = cv2.bitwise_not(canvas)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return canvas

cv2.namedWindow("Slide Bar")
cv2.resizeWindow("Slide Bar", 500, 500)
cv2.createTrackbar("Hue_Min", "Slide Bar", 0, 180, empty)
cv2.createTrackbar("Hue_Max", "Slide Bar", 180, 180, empty)
cv2.createTrackbar("Sat_Min", "Slide Bar", 0, 255, empty)
cv2.createTrackbar("Sat_Max", "Slide Bar", 90, 255, empty)
cv2.createTrackbar("Val_Min", "Slide Bar", 250, 255, empty)
cv2.createTrackbar("Val_Max", "Slide Bar", 255, 255, empty)
cv2.createTrackbar("Out", "Slide Bar", 75, 200, empty)
# cv2.createTrackbar("In", "Slide Bar", 950, 1000, empty)

cap = cv2.VideoCapture("White1.mp4")
success, img = cap.read()
img = cv2.resize(img, None, fx=.6, fy=.6, interpolation=cv2.INTER_CUBIC)

censor = full_censorship(img, "table")
cv2.imshow("Table2", censor)

cv2.waitKey(0)
while True:
    success, img = cap.read()
    img = cv2.resize(img, None, fx=.6, fy=.6, interpolation=cv2.INTER_CUBIC)
    copy = img.copy()

    # cv2.rectangle(img, (points[0]), (points[1]), (255, 0, 0), 1)

    hueMin = cv2.getTrackbarPos("Hue_Min", "Slide Bar")
    hueMax = cv2.getTrackbarPos("Hue_Max", "Slide Bar")
    satMin = cv2.getTrackbarPos("Sat_Min", "Slide Bar")
    satMax = cv2.getTrackbarPos("Sat_Max", "Slide Bar")
    valMin = cv2.getTrackbarPos("Val_Min", "Slide Bar")
    valMax = cv2.getTrackbarPos("Val_Max", "Slide Bar")
    outer = cv2.getTrackbarPos("Out", "Slide Bar")
    # inner = cv2.getTrackbarPos("In", "Slide Bar")

    cv2.imshow("Table1", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
