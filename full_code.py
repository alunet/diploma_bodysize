from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import math

#Функция для нахождения центральной точки
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#Получение входных аргументов
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in cm)")
args = vars(ap.parse_args())

#Загрузка изображения и его предобработка
image = cv2.imread(args["image"])
image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# поиск и выделение границ листа А4
edged = cv2.Canny(gray, 20, 20)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None
orig = None

for c in cnts:
	if cv2.contourArea(c) < 100:
		continue

	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	cv2.putText(orig, "{:.1f}cm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}cm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	cv2.imshow("Image", orig)
	break



#Прописываем путь к файлам с моделями
protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
 
#Считываем сеть в память
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

frameWidth = orig.shape[1]
frameHeight = orig.shape[0]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#Задаем размеры входного кадра
inWidth = 368
inHeight = 368

#Подготовим кадр к передаче в сеть
inpBlob = cv2.dnn.blobFromImage(orig, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
 
net.setInput(inpBlob)

output = net.forward()

H = output.shape[2]
W = output.shape[3]

#Зададим массив для хранения обнаруженных ключевых точек
points = []
for i in range(1,nPoints):

	probMap = output[0, i, :, :]

	minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

	x = (frameWidth * point[0]) / W
	y = (frameHeight * point[1]) / H
	if prob > threshold : 
		cv2.circle(orig, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
		cv2.putText(orig, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
 
		points.append((int(x), int(y)))
	else :
		points.append(None)
	if (i==13):
		break

#Проведем вычисление нужных значений для определения размеров тела(ширина плеч, длина рук, длина ног, длина туловища)
Shoulder = (points[4][0] - points[1][0])/pixelsPerMetric
Length = (points[7][1] - points[1][1])/pixelsPerMetric
Arm = (points[3][1]-points[1][1])/pixelsPerMetric
Leg = (points[12][1]-points[10][1])/pixelsPerMetric
print("Плечи",end=" ")
print(Shoulder)
print("Длина", end=" ")
print(Length)
print("Руки", end=" ")
print(Arm)
print("Ноги", end=" ")
print(Leg)

#В соответствии с таблицами размеров выведем рекомендованный данному человеку размер одежды
if (Shoulder < 44):
	print("Рекомендованный размер футболки - S")
elif (Shoulder < 48):
	print("Рекомендованный размер футболки - М")
else:
	print("Рекомендованный размер футболки - L")

if (Leg < 77):
	print("Рекомендованный размер штанов - S")
elif (Leg < 82):
	print("Рекомендованный размер штанов - М")
else:
	print("Рекомендованный размер штанов - L")


cv2.imshow("Image",orig)
cv2.waitKey(0)
cv2.destroyAllWindows()