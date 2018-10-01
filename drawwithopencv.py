import numpy as np
import cv2
import keras
from PIL import ImageGrab, Image

#globale variable
canvas = np.zeros([400,400,3],'uint8')
radius = 10
color = (255,255,255)
pressed = False

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('digitClassify.avi',fourcc, 20.0, (640,480))

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (3,3), 10)
    _, thresh = cv2.threshold(blured, 150, 255, cv2.THRESH_BINARY_INV)
    img11, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x1, y1, w1, h1 = cv2.boundingRect(contours[0])
    t = None
    if w1 > h1:
        t = w1
    else:
        t = h1
    t = t+10
    mask = np.zeros((t,t), dtype='uint8')
    x2 = int((t-w1)/2)
    y2 = int((t-h1)/2)
    mask[y2:y2+h1, x2:x2+w1] = blured[y1:y1+h1,x1:x1+w1]
    resize = cv2.resize(mask, (28,28))
    tpred = resize.reshape(1,28,28,1)
    #model = keras.models.load_model("mnist_digit.h5")
    #model = keras.models.load_model("mnist_digit_convolution.h5")
    model = keras.models.load_model('mnist_digit_convolution_w5e.h5')
    return np.argmax(model.predict(tpred))


#click function
def click(event, x, y, flag, param):
    #print("Event: ", event)
    #print("X: ", x, "  Y: ", y)
    #print("Flag: ", flag)
    #print("Param: ", param)
    global canvas, pressed, color
    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True
        cv2.circle(canvas,(x,y),radius,color,-1)
    elif event == cv2.EVENT_MOUSEMOVE and pressed:
        cv2.circle(canvas,(x,y),radius,color,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        pressed = False
        cv2.imwrite("digit.png", canvas)
        pred = preprocess_image(canvas)
        print("Predicted : " + str(pred))
    elif event == cv2.EVENT_RBUTTONDOWN:
        pressed = True
        color = (0, 0, 255)
        cv2.circle(canvas,(x,y),radius,color,-1)
    elif event == cv2.EVENT_RBUTTONUP:
        pressed = False
    #elif cv2.waitKey(0):
#        print(
     #   print("pressed A")
      #  color = (0,0,255)
    
cv2.namedWindow("canvas")
cv2.setMouseCallback("canvas", click)

while True:
    cv2.imshow("canvas", canvas)
    screen = np.array(ImageGrab.grab(bbox=(10,10,900,900)))
    #print(screen.shape)
    resized_screen = cv2.resize(screen, (640,480), Image.ANTIALIAS)
    #cv2.imshow("Screen", resized_screen)
    out.write(resized_screen)
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break
    if ch == ord('c'):
        canvas = canvas * 0

cv2.destroyAllWindows()
