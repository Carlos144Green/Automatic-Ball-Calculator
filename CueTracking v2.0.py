import cv2
import numpy as np
corners = np.zeros((2, 2), np.int)
kernel = np.ones([3, 3], np.uint8)
def empty(a):
   pass
def cue_finder(censored_stream):
    lines = cv2.HoughLinesP(censored_stream, 1, np.pi / 180, threshold=100, minLineLength=20, maxLineGap=75)
    if lines is not None:
        if np.size(lines, 1) >= 2:
            x10, y10, x20, y20 = lines[0]
            x11, y11, x21, y21 = lines[1]
            x1, y1, x2, y2 = (x10 + x11) / 2, (y10 + y11) / 2, (x20 + x21) / 2, (y20 + y21) / 2
        else:
            x1, y1, x2, y2 = lines[0, 0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** .5
        if length <= 1000:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            length1 = ((x1 - fakeCueBall[0]) ** 2 + (y1 - fakeCueBall[1]) ** 2) ** .5
            length2 = ((x2 - fakeCueBall[0]) ** 2 + (y2 - fakeCueBall[1]) ** 2) ** .5
            if length1 <= length2:
                cv2.circle(img, (x1, y1), 3, (200, 100, 100), 2)
            else:
                cv2.circle(img, (x2, y2), 3, (200, 100, 100), 2)
        pastLine.append([(x1, y1),(x2, y2)])
    else:
        pastLine.append(None)
    if len(pastLine) > 10:
        pastLine.pop(0)

    if not all(p is None for p in pastLine):  # Checks if completely empty
        # print("pastLine: ", pastLine)
        noNull = []
        for check in pastLine:
            # print("check: ", check)
            if check is not None:
                noNull.append(check)

        # print("noNull", noNull)
        print("mean: {}".format(np.mean(noNull, axis=0), ))
        predictLine = np.mean(noNull, axis=0)
        predictLine = np.round(predictLine[:, :]).astype("int")  # Make all balls x&y int
        print("preLine: ", predictLine)
        cv2.line(img, (predictLine[0][0], predictLine[0][1]), (predictLine[1][0], predictLine[1][1]), (0, 0, 0), 2)

cap = cv2.VideoCapture("White1.mp4")
success, img = cap.read()
img = cv2.resize(img, None, fx=.6, fy=.6, interpolation=cv2.INTER_CUBIC)


bbox = cv2.selectROI("Tracking", img, True)
cv2.destroyWindow("Tracking")
# gEmpty = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# fake cue ball
fakeCueBall = (500, 250)
pastLine = [None, None, None]
while True:
    success, img = cap.read()
    timer = cv2.getTickCount()
    img = cv2.resize(img, None, fx=.6, fy=.6, interpolation=cv2.INTER_CUBIC)
    ##################################################
    gIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # iSpy = cv2.absdiff(gIMG, gEmpty)
    # diff = cv2.bitwise_not(iSpy)

    edges = cv2.Canny(gIMG, 50, 150, apertureSize=3)
    blank = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    thicc = 28
    border = cv2.rectangle(blank, (bbox[0]-thicc, bbox[1]-thicc), (bbox[0] + bbox[2]+thicc, bbox[1] + bbox[3]+thicc), (255, 255, 255), thicc*2)
    cv2.rectangle(border, (575, 28), (735, 110), (255, 255, 255), -1)
    border = cv2.bitwise_not(border)
    border = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)


    test = cv2.bitwise_and(edges, border)
    cv2.rectangle(img, (bbox[0]-thicc, bbox[1]-thicc), (bbox[0] + bbox[2]+thicc, bbox[1] + bbox[3]+thicc), (255, 255, 255), thicc*2)
    # cv2.circle(img, fakeCueBall, 10, (255,0,20),1)

    cue_finder(test)

    ##################################################
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_COMPLEX, .7, (100, 0, 255), 2)
    ##################################################
    cv2.imshow("Table1", img)

    # cv2.imshow("Table2", test)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break