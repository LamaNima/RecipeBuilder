from django.shortcuts import render
from .models import user
# Create your views here.

def home_page(request):
    users = user.objects.all().order_by('-date')
    return render(request,'main/main.html',{'users':users})



    