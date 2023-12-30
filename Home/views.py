from django.shortcuts import render,redirect
from .forms import UserAddForm
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from django.contrib.auth.models import User, Group
from .models import Student,Attendence, SemMark

import cv2
import os

from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib


datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('Home\haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


# Create your views here.
def Index(request):
    att = Attendence.objects.filter(date = date.today())
    context = {
        "att":att
    }
    return render(request,"index.html",context)

def SignIn(request):
    if request.method == "POST":
        username = request.POST['uname']
        password = request.POST['pswd']
        user1 = authenticate(request, username = username , password = password)
        
        if user1 is not None:
            
            request.session['username'] = username
            request.session['password'] = password
            login(request, user1)
            return redirect('Index')
        
        else:
            messages.info(request,'Username or Password Incorrect')
            return redirect('SignIn')
    return render(request,"login.html")

def SignUp(request):
    form = UserAddForm()
    if request.method == "POST":
        form = UserAddForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            email = form.cleaned_data.get("email")
            if User.objects.filter(username = username).exists():
                messages.info(request,"Username Exists")
                return redirect('SignUp')
            if User.objects.filter(email = email).exists():
                messages.info(request,"Email Exists")
                return redirect('SignUp')
            else:
                new_user = form.save()
                new_user.save()
                
                messages.success(request,"User Created")
                return redirect('AdminHome')
            
    return render(request,"register.html",{"form":form})

def SignOut(request):
    logout(request)
    return redirect('SignIn')

# student add data function

def AddStudent(request):
    if request.method == "POST":
        sname = request.POST["sname"]
        rnum = request.POST["rnum"]
        if Student.objects.filter(student_name = sname, roll = rnum).exists():
            messages.info(request,"Same Name and roll number exists")
            return redirect("Index")
        else:
            stud = Student.objects.create(student_name=sname, roll=rnum )
            stud.save()
        return redirect("MarkAttendence",stud.id)

#Extract face    
def extract_faces(img):
    if img is not None and img.size != 0:  # Check if the image is not None and not empty
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return [] 

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl') 

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l

def totalreg():
    return len(os.listdir('static/faces')) 

def MarkAttendence(request,pk):
    student = Student.objects.get(id = pk)
    if request.method == "POST":
        userimagefolder = 'static/faces/'+student.student_name+'_'+str(student.roll)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        cap = cv2.VideoCapture(0)
        i,j = 0,0
        while 1:
            _,frame = cap.read()
            faces = extract_faces(frame)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if j%10==0:
                    name = student.student_name+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                    i+=1
                j+=1
            if j==500:
                break
            cv2.imshow('Adding new User',frame)
            if cv2.waitKey(1)==27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print('Training Model')
        train_model()
        names,rolls,times,l = extract_attendance()   
        if totalreg() > 0 :
            messages.info(request,"Face Added1")
            return redirect('Index')
        else:
            messages.info(request,"Face added")
            return redirect("Index")

    context = {"student":student}

    return render(request,"markattendence.html",context)

def AttendenceExits(request):
    messages.info(request,"You already marked your attendence")
    return redirect("Index")

def AttendenceMarked(request):
    messages.info(request,"Your attendence marked")
    return redirect("Index")

def FaceNotMatch(request):
    messages.info(request,"The Face is not matching")
    return redirect("Index")


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    try:
        student = Student.objects.get(student_name = username, roll = userid)
        if Attendence.objects.filter(student = student, date = date.today()).exists():
            return redirect("AttendenceExits")
        else:
            Atte = Attendence.objects.create(student = student)
            Atte.save()
        cap.release()
        cv2.destroyAllWindows()
        
        return redirect("AttendenceMarked")
    except:
        cap.release()
        cv2.destroyAllWindows()
        # messages.info(request,"The Face is not matching")
        return redirect("FaceNotMatch")
    
    # df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    # if str(userid) not in list(df['Roll']):
    #     with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
    #         f.write(f'\n{username},{userid},{current_time}')


def TakeAttendence(request):
    if request.method == "POST":

        if 'face_recognition_model.pkl' not in os.listdir('static'):
            messages.info(request,'There is no trained model in the static folder. Please add a new face to continue.')
            return redirect('Index')
        cap = cv2.VideoCapture(0)
        ret = True
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale frame
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1,-1))[0]
                cv2.putText(frame,f'{identified_person}',(x + 6, y - 6),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2)
                add_attendance(identified_person)
                # img, name, , , 1, (255, 255, 255), 2

            # Display the resulting frame
            cv2.imshow('Attendance Check', frame)
            cv2.putText(frame,'hello',(30,30),cv2.FONT_HERSHEY_COMPLEX,2,(255, 255, 255))
            break
            
        # Wait for the user to press 'q' to quit
            if cv2.waitKey(1)==27 & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        names,rolls,times,l = extract_attendance()    
        return redirect("Index") 

        messages.info(request,"Attendence Marked")
        return redirect("Index")
    
    return redirect("Index")


def AllAttendence(request):
    att = Attendence.objects.all()
    context = {
        "att":att
    }
    return render(request,'attendence.html',context)


def Allstudents(request):
    stu = Student.objects.all()
    context = {
        "stu":stu
    }
    return render(request,'students.html',context)

def StudentSingle(request,pk):
    stu = Student.objects.get(id=pk)

    try:
        att = Attendence.objects.filter(student = stu)
    except:
        pass
    try:
        mark = SemMark.objects.filter(student = stu)
    except:
        pass
    if request.method == "POST":
        sem = request.POST["sem"]
        mark = request.POST["mark"]
        semmark = SemMark.objects.create(student = stu, semester = sem,mark =mark)
        semmark.save()
        return redirect("StudentSingle",pk)
    context = {
        "stu":stu,
        "att":att,
    }
    try:
        context["mark"] = mark
    except:
        pass
    return render(request,"studentsingle.html",context)

def SearchStudent(request):
    if request.method == "POST":
        key = request.POST['search']
        stu = Student.objects.filter(student_name__contains = key)
        context = {
            "stu":stu
        }
    return render(request,'searchres.html',context)
