import cv2
import numpy as np
from keras.models import load_model


model = load_model('model_num_detection.h5')

img = "try9.jpeg"
image = cv2.imread(img, cv2.IMREAD_COLOR)


gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
image = cv2.fastNlMeansDenoisingColored(image,None,20,20,7,21)
ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

threshold_area_min = (image.shape[1]*image.shape[0])/900
threshold_area_max = (image.shape[1]*image.shape[0])/150

contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

ans = []

for cnt in contours:
    area = cv2.contourArea(cnt) 
    if area >= 100: 
        print(area)
    if  threshold_area_max >= area >= threshold_area_min: 
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        digit = th[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))

        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0

        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)

        data = str(final_pred)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ans.append(final_pred)

print(ans)
