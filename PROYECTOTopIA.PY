import cv2
from matplotlib import pyplot as plt


captura = cv2.VideoCapture('./placas1.mp4')

while (captura.isOpened()):
    ret,frame = captura.read()
    
    escalado = 60
    width = int(frame.shape[1]*escalado/200)
    height = int(frame.shape[0]*escalado/200)
    
    imagen = cv2.resize(frame,(width,height),interpolation = cv2.INTER_AREA)
    
    filtro1 = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    filtro2 = cv2.GaussianBlur(filtro1 ,(7,7),0 )
    filtro3 = cv2.Canny(filtro2 ,15, 200)
    canny = cv2.dilate(filtro3, None, iterations=1)
    
    cnts,_ = cv2.findContours(canny,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        epsilon = 0.09*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        if len(approx) == 4 and area > 5000:
           #print('area= ', area)
            cv2.drawContours(imagen,[c], 0, (255,0,0),2)

            placa =imagen[y:y+h,x:x+w]
         #  cv2.imwrite('placa.jpg',placa)
            cv2.imshow('placa encontrada',placa)
            break

           
    cv2.imshow('GRISES',filtro1)
    cv2.imshow('GAUSSIANO',filtro2)    
    cv2.imshow('canny',filtro3)
    cv2.imshow('imagen',imagen)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    
captura.release()
cv2.destroyAllWindows()