import cv2

img = cv2.imread('images/ex2.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

body = cv2.CascadeClassifier('body.xml')

results = body.detectMultiScale(img_gray, scaleFactor=2, minNeighbors=3)

for (x, y, w, h) in results:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)

if results.all() != 0:
    print(f'Высота в пикселях:', results[0][2])

cv2.imshow('Result', img)
cv2.waitKey(0)



