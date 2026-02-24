from picamera2 import Picamera2, Preview
#import keyboard
from time import sleep
import cv2



picam = Picamera2()

camera_config = picam.create_preview_configuration()
picam.configure(camera_config)
picam.start_preview()
picam.start()



while True:
    frame = picam.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("LiveCameraStream", frame_bgr)
    
    if cv2.waitKey(1) == ord('q'):
        picam.capture_file('image.jpg')
        break
cv2.destroyAllWindows()
picam.stop()	
