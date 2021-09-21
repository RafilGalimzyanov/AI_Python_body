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



'''img = cv2.imread('images/OpenCV.jpg')
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 3))
img = cv2.Canny(img, 200, 200) # Находит контуры (перевод в бинарное представление
cv2.imshow('Result', img) #Показать
cv2.waitKey(0) #Ждать в миллисек (0 - бесконечно)'''

'''cap = cv2.VideoCapture(0) # аргумент 0 - фронтальная камера, либо путь к видеофайлу

while True:
    success, img = cap.read() # suc - true or false, img - изображение
    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'): # кнопка q для выключения
        break'''