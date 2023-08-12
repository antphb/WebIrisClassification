from django.shortcuts import redirect, render
from django.views import View
from django.http import JsonResponse
import pandas as pd
import joblib
# Create your views here.

model_svm=joblib.load('app/static/model/model_svm.pkl')
df=pd.read_csv('app/static/file/iris.csv')

lable_path_image={
    2: "Iris_virginica.jpg",
    1: "Iris_versicolor.jpg",
    0: "Iris_setosa.jpg"
}
class index(View):
    def get(self,request):
        return render(request,'index.html')

class Predict(View):
    def get(self,request):
        return redirect(request.META.get('HTTP_REFERER'))
    
    def post(self,request):
        sepal_length=request.POST['sepal_length']
        sepal_width=request.POST['sepal_width']
        petal_length=request.POST['petal_length']
        petal_width=request.POST['petal_width']
        prediction=model_svm.predict([[sepal_length,sepal_width,petal_length,petal_width]])
        path_image=lable_path_image[prediction[0]]
        name_image=" ".join(path_image.split('.')[0].split('_'))
        return JsonResponse({'path_image': path_image,'name_image': name_image})
    
class get_data(View):
    def get(self,request):
        df_setosa=df[df['class']=='Iris-setosa']
        df_versicolor=df[df['class']=='Iris-versicolor']
        df_virginica=df[df['class']=='Iris-virginica']
        dict_data={
            'setosa':{
                'sepal_length': df_setosa.sepal_length.tolist(),
                'sepal_width': df_setosa.sepal_width.tolist(),
                'petal_length': df_setosa.petal_length.tolist(),
                'petal_width': df_setosa.petal_width.tolist(),
            },
            'versicolor':{
                'sepal_length': df_versicolor.sepal_length.tolist(),
                'sepal_width': df_versicolor.sepal_width.tolist(),
                'petal_length': df_versicolor.petal_length.tolist(),
                'petal_width': df_versicolor.petal_width.tolist(),
            },
            'virginica':{
                'sepal_length': df_virginica.sepal_length.tolist(),
                'sepal_width': df_virginica.sepal_width.tolist(),
                'petal_length': df_virginica.petal_length.tolist(),
                'petal_width': df_virginica.petal_width.tolist(),
            }
        }
        return JsonResponse(dict_data)