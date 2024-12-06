import cv2
import RPi.GPIO as GPIO
import time

LED_PIN = 18
SERVO_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(7.5)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def set_servo_angle(angle):
    duty = 2.5 + (angle / 18.0)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.2)

camera = cv2.VideoCapture(0)
current_angle = 90
set_servo_angle(current_angle)

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Не удалось получить кадр с камеры")
            break
        
        if detect_face(frame):
            print("Лицо обнаружено")
            GPIO.output(LED_PIN, GPIO.HIGH)
        else:
            print("Лицо не обнаружено")
            GPIO.output(LED_PIN, GPIO.LOW)
            current_angle += 15
            if current_angle > 180:
                current_angle = 0
            set_servo_angle(current_angle)
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Завершение программы")

finally:
    camera.release()
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()
