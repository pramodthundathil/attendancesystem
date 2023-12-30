from django.urls import path
from .import views


urlpatterns = [
    path("Index",views.Index,name="Index"),
    path("",views.SignIn,name="SignIn"),
    path("SignUp",views.SignUp,name="SignUp"),
    path("SignOut",views.SignOut,name="SignOut"),
    path("AddStudent",views.AddStudent,name="AddStudent"),
    path("MarkAttendence/<int:pk>",views.MarkAttendence,name="MarkAttendence"),
    path("TakeAttendence",views.TakeAttendence,name="TakeAttendence"),
    path("AllAttendence",views.AllAttendence,name="AllAttendence"),
    path("Allstudents",views.Allstudents,name="Allstudents"),
    path("SearchStudent",views.SearchStudent,name="SearchStudent"),
    path("AttendenceExits",views.AttendenceExits,name="AttendenceExits"),
    path("AttendenceMarked",views.AttendenceMarked,name="AttendenceMarked"),
    path("FaceNotMatch",views.FaceNotMatch,name="FaceNotMatch"),
    path("StudentSingle/<int:pk>",views.StudentSingle,name="StudentSingle"),
 




]

