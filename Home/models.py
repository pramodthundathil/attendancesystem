from django.db import models

class Student(models.Model):
    student_name = models.CharField(max_length=255)
    roll = models.IntegerField()
    class_name = models.CharField(max_length = 255,null= True, blank = True)

class Attendence(models.Model):
    student  = models.ForeignKey(Student,on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)

class SemMark(models.Model):
    student = models.ForeignKey(Student,on_delete = models.CASCADE)
    semester = models.CharField(max_length=255)
    mark = models.FloatField()
    status = models.BooleanField(default = True)
    
