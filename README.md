algoritmo1_seminario
Algoritmo preliminar de prototipo de aplicación móvil para el aporte de lectura de clientes con tarifa básica para una empresa de distribución eléctrica

Inicio de algoritmo
se cargan la librerias.

    import cv2
    import numpy as np
    import skimage
    from paddleocr import PaddleOCR,draw_ocr
    
#Se genera una matriz de orden 2 para el eje X y eje Y, cada punto posee un color distinto, estos son ordenados en base a una jerarquia para dibujar contornos.

    def ordenar_puntos(puntos):
        n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()

        y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])

        x1_order = y_order[:2]
        x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])

        x2_order = y_order[2:4]
        x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])

        return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

    image = cv2.imread('C:/Users/juliu/Desktop/OCR2/medidor_luz/505_3.jpg')

    ocr = PaddleOCR(use_angle_cls=True, lang='german')

    resize = cv2.resize(image, (600,800))

    gray = cv2.cvtColor(resize, cv2.COLOR_RGB2GRAY)
    median = cv2.medianBlur(gray, 5)

    sci = skimage.exposure.equalize_hist(median)
    sci2 = skimage.exposure.adjust_gamma(sci, 1, 1) #3

    #rescatar = skimage.exposure.rescale_intensity(sci2, out_range=(120, 150)).astype(np.int8)

    sci3 = skimage.img_as_ubyte(sci2)

    #cv2.imshow('sci2', sci3)
    canny = cv2.Canny(sci3, 70, 250) #70, 250
    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.dilate(canny,kernel,iterations=1)

    cnts, h = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    for c in cnts:
        epsilon = 0.01*cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)

        #print('approx', approx)
        if len(approx) == 4:

            puntos = ordenar_puntos(approx)

            pts1 = np.float32(puntos)
            pts2 = np.float32([[0,0],[270,0],[0,70],[270,70]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(gray, M , (270, 70))
            dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

            dst2 = cv2.warpPerspective(resize, M, (270, 70))

            lower_red = np.array([50,50,70])
            upper_red = np.array([60,60,140])
            rojo = cv2.inRange(dst2, lower_red, upper_red)
            res = cv2.bitwise_and(dst2, dst2, mask=rojo)

            if res.max() > 0:

                cv2.drawContours(resize, [approx], 0, (0,255,255), 2)

                cv2.circle(resize, tuple(puntos[0]), 7, (255,0,0), 2)
                cv2.circle(resize, tuple(puntos[1]), 7, (0,255,0), 2)
                cv2.circle(resize, tuple(puntos[2]), 7, (255,100,0), 2)
                cv2.circle(resize, tuple(puntos[3]), 7, (255,255,0), 2)

                resize_as_double = cv2.resize(dst, (dst.shape[1]*2,dst.shape[0]*2))

                ret,thresh1 = cv2.threshold(resize_as_double,103,255,cv2.THRESH_TOZERO) #103, 255

                resultado_paddle = ocr.ocr(thresh1, det=False, rec=True, cls=True)

                text = '' #PADDLE
                for pred in resultado_paddle:
                    print(pred[-2]) 
                    text = text + '\n' + pred[-2]
                cv2.putText(resize, pred[-2], (40,40), 1, 2.0, (0,0,255), 3 )

                cv2.imshow('th', thresh1)
                cv2.waitKey(0)

    cv2.imshow('resize', resize)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
