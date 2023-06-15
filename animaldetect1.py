import cv2
import requests

animalNames = []
animal = "Contents.names"
classNames = []
classFile = "coco.names"
#classFile = "Contents.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

with open(animal, "rt") as f:
   animalNames = f.read().rstrip("\n").split("\n")


configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath ="frozen_inference_graph.pb"


net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



def telegram_bot_sendtext(bot_message):
    bot_token="5213361505:AAGjyPVvdAg6uHXKs9VHwd4NbW6nq0yfGzE"
    bot_chatID="1051081941"
    send_text='https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id='+ bot_chatID +  '&parse_mode=MarkdownV2&text=' + bot_message
    response= requests.get(send_text)
    return response.json

msg=''

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)

    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if (draw):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    Test=True#[True if x == classNames[classId - 1] else False for x in animalNames]
                    #if(classNames[classId - 1]):
                     #   print(classNames[classId - 1])
                    if(Test):
                        msg='Animal detected- '+ classNames[classId - 1]
                        telegram_bot_sendtext(msg)

    return img, objectInfo


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)#('Videos/Bird-1.mp4')#   

    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1240, 680))
        result, objectInfo = getObjects(img, 0.45, 0.2)
        cv2.imshow("Output", result)

        if (cv2.waitKey(3) & 0xFF == ord('q')):
            break