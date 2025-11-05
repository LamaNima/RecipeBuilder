from django.shortcuts import render
from .utils.model_utils import get_recipes
from django.core.files.storage import default_storage

from django.http import JsonResponse
from PIL import Image
import os
# Create your views here.

def index(request):
    return render(request,'main/index.html')

def upload_image(request):
    return render(request, 'main/upload.html')

    
def analyze_food(request):
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        
        image_path = default_storage.save(f"uploads/{image.name}", image)
        full_path = os.path.join(default_storage.location, image_path)

        try:
            # Call your model here
            img = Image.open(full_path).convert("RGB")
            result = get_recipes(img,top_k=1,num_recipes=5)

            return JsonResponse(result, safe=False)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
