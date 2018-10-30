import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist

path="C:/Users/hp/Desktop/AI/Udemy CV/shape_predictor_68_face_landmarks.dat"
predictor=dlib.shape_predictor(path)
detector=dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects=detector(im,1)#image and no.of rectangles to be drawn
    if len(rects)>1:
        print("Toomanyfaces")
        return np.matrix([0])
    if len(rects)==0:
        print("Toofewfaces")
        return np.matrix([0])
    return np.matrix([[p.x,p.y] for p in predictor(im,rects[0]).parts()]) 

  
def place_landmarks(im,landmarks):
    im=im.copy()
    for idx,point in enumerate(landmarks):
        pos=(point[0,0],point[0,1])
        cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.3,color=(0,255,255))
        cv2.circle(im,pos,3,color=(0,255,255))
    return im 


def upper_lip(landmarks):
    top_lip=[]
    for i in range(50,53):
        top_lip.append(landmarks[i])
    for j in range(61,64):
        top_lip.append(landmarks[j])
    top_lip_point=(np.squeeze(np.asarray(top_lip)))
    top_mean=np.mean(top_lip_point,axis=0)
    
    return int(top_mean[1])
    
        
def low_lip(landmarks):
    lower_lip=[]
    for i in range(65,68):
        lower_lip.append(landmarks[i])
    for j in range(56,59):
        lower_lip.append(landmarks[j])
    lower_lip_point=(np.squeeze(np.asarray(lower_lip)))
    lower_mean=np.mean(lower_lip_point,axis=0)
    return int(lower_mean[1])


def left_lash(landmarks):
    p1,p2,p3,p4,p5,p6=landmarks[36],landmarks[37],landmarks[38],landmarks[39],landmarks[40],landmarks[41]
    a=dist.euclidean(p2,p6)
    b=dist.euclidean(p1,p4)
    c=dist.euclidean(p3,p5)
    ear1=(a+c)/(2.0*b)
    return ear1

def right_lash(landmarks):
    q1,q2,q3,q4,q5,q6=landmarks[42],landmarks[43],landmarks[44],landmarks[45],landmarks[46],landmarks[47]
    d=dist.euclidean(q2,q6)
    e=dist.euclidean(q1,q4)
    f=dist.euclidean(q3,q5)
    ear2=(d+f)/(2.0*e)
    return ear2  
        
        
               
def decision1(image):
    landmarks=get_landmarks(image)
    if(landmarks.all()==[0]):
        return -10
    
    top_lip=upper_lip(landmarks)
    lower_lip=low_lip(landmarks)
    distance=abs(top_lip-lower_lip)
    return distance

               
def decision2(image):
    landmarks=get_landmarks(image)
    if(landmarks.all()==[0]):
        return -10
    left=left_lash(landmarks)
    right=right_lash(landmarks)
    ear=(left+right)/2.0
    return ear
    
    

    
    
        
              
cap=cv2.VideoCapture(0)
yawns=0
blink=0
ear_threshold=0.27
no_of_frames=3
counter=0
while(True):
    
    ret,frame=cap.read()
    if(ret==True):

        landmarks=get_landmarks(frame)
        
        if(landmarks.all()!=[0]):
            l1=[]
            for k in range(48,60):
                l1.append(landmarks[k])
            l2=np.asarray(l1)
            lips=cv2.convexHull(l2)
            cv2.drawContours(frame, [lips], -1, (0, 255, 0), 1)
        
            l3=[]
            for x in range(36,42):
                l3.append(landmarks[x])
            l4=np.asarray(l3)
            left=cv2.convexHull(l4)
            cv2.drawContours(frame, [left], -1, (0, 255, 0), 1)
        
            l5=[]
            for y in range(42,48):
                l5.append(landmarks[y])
            l6=np.asarray(l5)
            right=cv2.convexHull(l6)
            cv2.drawContours(frame, [right], -1, (0, 255, 0), 1)        
        
        
        

        distance1=decision1(frame)
        EAR=decision2(frame)
        
        if(distance1>21):   #Set distance according to your convenience
            yawns=yawns+1
        cv2.putText(frame,"Yawn Count: "+str(yawns),(50,100),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,color=(0,0,255))
        if(EAR<=ear_threshold):
            counter=counter+1
        if(EAR>ear_threshold):
            if(counter>=no_of_frames):
                blink=blink+1
            counter=0    
           
        cv2.putText(frame,"Blink Count: "+str(blink),(500,100),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,255)) 
        cv2.putText(frame,"EAR count: "+str(EAR),(500,200),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,255))
            
        cv2.imshow("Dozing Detection",frame)
        
        if cv2.waitKey(1)==13:
            break
    else:
        continue
    
cap.release()
cv2.destroyAllWindows()    
    
    
    
        
               
        
        
    
