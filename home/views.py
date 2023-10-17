import imghdr

from django.shortcuts import render, HttpResponse, redirect
import joblib
from .forms import PredictionForm, PredictionForm_dtree
from datetime import datetime
from .models import Prediction, PredictionResult, PredictionResult2, PredictionResult3
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import user_passes_test

from django.contrib.auth.decorators import login_required
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


# from sklearn.externals import joblib


def sign(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        if pass1 != pass2:
            messages.warning(request, "your passwords doesn't matching")
        else:
            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()
            return redirect('loginp')
        print(uname, email, pass1, pass2)
    return render(request, 'sign.html')


def loginp(request):
    if request.method == "POST":
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        user = authenticate(username=username, password=pass1)
        print(username, pass1)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.warning(request, "Username or password is incorrect")

    return render(request, 'loginp.html')


def logoutp(request):
    logout(request)
    return redirect("loginp")


def base(request):
    return  render(request, 'base.html')


def mainhome(request):
    if request.method == 'POST':
        # feature1 = float(request.POST['feature1'])
        # feature2 = float(request.POST['feature2'])
        # # feature1 = request.POST.get('feature1')
        # # feature2 = request.POST.get('feature2')
        # model = joblib.load('home/static/ml_linear_model.pkl')
        # prediction = model.predict([[feature1, feature2]])

        feature1 = float(request.POST['feature1'])
        model = joblib.load('home/static/ml_dtree_model.pkl')
        prediction = model.predict([[feature1]])
        return render(request, 'mainhome.html', {'prediction': prediction[0]})

    return render(request, 'mainhome.html')


# def predict_demand_supply_dtree(request):
#     if request.method == 'POST':
#         form_dtree = PredictionForm_dtree(request.POST)
#         if form_dtree.is_valid():
#             input_date_dtree = form_dtree.cleaned_data['date']
#             input_year = input_date_dtree.year
#             input_month = input_date_dtree.month
#             input_day = input_date_dtree.day
#             dtree_demand_model = joblib.load('home/static/ml_dtree_demand_model.pkl')
#             dtree_supply_model = joblib.load('home/static/ml_dtree_supply_model.pkl')
#             input_features = [[input_year, input_month, input_day]]
#             demand_prediction_dtree = dtree_demand_model.predict(input_features)
#             supply_prediction_dtree = dtree_supply_model.predict(input_features)
#             context = {
#                 'input_date': input_date_dtree,
#                 'demand_prediction': demand_prediction_dtree[0],
#                 'supply_prediction': supply_prediction_dtree[0],
#             }
#
#             return render(request, 'prediction_results.html', context)
#     else:
#         form_dtree = PredictionForm_dtree()
#
#     return render(request, 'predict.html', {'form': form_dtree})
# @login_required
# @user_passes_test(lambda u: u.has_perm("home.view_prediction result"))
def predict_demand_supply_dtree(request):
    if request.method == 'POST':
        input_date_dtree_str = request.POST['date']
        try:
            prediction = Prediction.objects.create(date=input_date_dtree_str)  # Create a new Prediction object
        except ValueError:
            return HttpResponse('Invalid date format or other error occurred', status=400)
        # input_date_dtree = prediction.cleaned_data['date']
        input_date_dtree = datetime.strptime(input_date_dtree_str, '%Y-%m-%d').date()
        input_year = input_date_dtree.year
        input_month = input_date_dtree.month
        input_day = input_date_dtree.day
        # Load models and make predictions
        dtree_demand_model = joblib.load('home/static/ml_dtree_demand_model.pkl')
        dtree_supply_model = joblib.load('home/static/ml_dtree_supply_model.pkl')
        input_features = [[input_year, input_month, input_day]]
        demand_prediction_dtree = dtree_demand_model.predict(input_features)
        supply_prediction_dtree = dtree_supply_model.predict(input_features)

        prediction_result = PredictionResult(
            input_date=input_date_dtree,
            demand_prediction=demand_prediction_dtree[0],
            supply_prediction=supply_prediction_dtree[0]
        )
        prediction_result.save()

        context = {
            'input_date': input_date_dtree,
            'demand_prediction': demand_prediction_dtree[0],
            'supply_prediction': supply_prediction_dtree[0],
        }

        return render(request, 'prediction_results.html', context)
    else:
        return render(request, 'prediction_results.html')


def predictResult(request):
    return render(request, 'prediction_results.html')


# def pcw(request):
#     if request.method == 'POST':
#         input_category_dtree = request.POST['categorySelect']
#         input_date_dtree_str = request.POST['date']
#         try:
#             prediction = Prediction.objects.create(date=input_date_dtree_str)  # Create a new Prediction object
#         except ValueError:
#             return HttpResponse('Invalid date format or other error occurred', status=400)
#         # input_date_dtree = prediction.cleaned_data['date']
#         # input_category_dtree =
#         input_date_dtree = datetime.strptime(input_date_dtree_str, '%Y-%m-%d').date()
#         input_year = input_date_dtree.year
#         input_month = input_date_dtree.month
#         input_day = input_date_dtree.day
#         # Load models and make predictions
#         dtree_demand_model = joblib.load('home/static/ml_dtree_demand_model.pkl')
#         dtree_supply_model = joblib.load('home/static/ml_dtree_supply_model.pkl')
#         input_features = [[input_year, input_month, input_day]]
#         demand_prediction_dtree = dtree_demand_model.predict(input_features)
#         supply_prediction_dtree = dtree_supply_model.predict(input_features)
#
#         prediction_result = PredictionResult(
#             input_category=input_category_dtree,
#             input_date=input_date_dtree,
#             demand_prediction=demand_prediction_dtree[0],
#             supply_prediction=supply_prediction_dtree[0]
#         )
#         prediction_result.save()
#
#         context = {
#             'input_category': input_category_dtree,
#             'input_date': input_date_dtree,
#             'demand_prediction': demand_prediction_dtree[0],
#             'supply_prediction': supply_prediction_dtree[0],
#         }
#
#         return render(request, 'prediction_results2.html', context)
#     else:
#         return render(request, 'prediction_results2.html')
#     # return render(request, 'pcw.html')
#
#

# @user_passes_test(lambda u: u.has_perm("home.view_prediction result2"))
def pcw(request):
    if request.method == 'POST':
        category_choices = ['Health and beauty', 'Electronic accessories', 'Home and lifestyle', 'Fashion accessories',
                            'Food and beverages', 'Sports and travel']

        input_category_dtree = request.POST.get('categorySelect')

        if input_category_dtree not in category_choices:
            return HttpResponse('Invalid category selected', status=400)

        input_date_dtree_str = request.POST.get('date')

        try:
            prediction = Prediction.objects.create(date=input_date_dtree_str)  # Create a new Prediction object
        except ValueError:
            return HttpResponse('Invalid date format or other error occurred', status=400)

        input_date_dtree = datetime.strptime(input_date_dtree_str, '%Y-%m-%d').date()
        input_year = input_date_dtree.year
        input_month = input_date_dtree.month
        input_day = input_date_dtree.day

        # Define the path to the model based on the selected category
        model_path = f'home/static/Category_{input_category_dtree}_demand_model.pkl'

        try:
            dtree_demand_model = joblib.load(model_path)
        except FileNotFoundError:
            return HttpResponse('Model not found for the selected category', status=400)

        input_features = [[input_year, input_month, input_day]]
        demand_prediction_dtree = dtree_demand_model.predict(input_features)

        # Similar modifications for supply model path
        supply_model_path = f'home/static/Category_{input_category_dtree}_supply_model.pkl'
        try:
            dtree_supply_model = joblib.load(supply_model_path)
        except FileNotFoundError:
            return HttpResponse('Supply Model not found for the selected category', status=400)

        supply_prediction_dtree = dtree_supply_model.predict(input_features)

        Demand_Graph = "static/image/Demand_plot.jpg"
        Supply_Graph = "static/image/Supply_plot.jpg"
        Demand_And_Supply_Graph = "static/image/DemandAndSupply_plot.jpg"
        Dtree_Scatter_Graph = "static/image/dtree_scatter_plot.png"

        prediction_result2 = PredictionResult2(
            input_category=input_category_dtree,
            input_date=input_date_dtree,
            demand_prediction=demand_prediction_dtree[0],
            supply_prediction=supply_prediction_dtree[0],
            Demand_Graph=Demand_Graph,
            Supply_Graph=Supply_Graph,
            Demand_And_Supply_Graph=Demand_And_Supply_Graph,
            Dtree_Scatter_Graph=Dtree_Scatter_Graph,
        )
        prediction_result2.save()

        context = {
            'input_category': input_category_dtree,
            'input_date': input_date_dtree,
            'demand_prediction': demand_prediction_dtree[0],
            'supply_prediction': supply_prediction_dtree[0],
            'Demand_Graph': Demand_Graph,
            'Supply_Graph': Supply_Graph,
            'Demand_And_Supply_Graph': Demand_And_Supply_Graph,
            'Dtree_Scatter_Graph': Dtree_Scatter_Graph,
        }

        return render(request, 'prediction_results2.html', context)
    else:
        return render(request, 'prediction_results2.html')


def predictResult2(request):
    return render(request, 'prediction_results2.html')


# @user_passes_test(lambda u: u.has_perm("home.view_prediction result3"))
def pcacw(request):
    if request.method == 'POST':
        category_choices = ['Health and beauty', 'Electronic accessories', 'Home and lifestyle', 'Fashion accessories',
                            'Food and beverages', 'Sports and travel']

        city_choices = ['Delhi', 'Pune', 'Mumbai', 'Kanpur', 'Surat', 'Chennai', 'Bangalore',
                        'Kolkata', 'Ahmedabad', 'Hyderabad']

        input_city_dtree = request.POST.get('citySelect')
        input_category_dtree = request.POST.get('categorySelect')

        if input_city_dtree not in city_choices:
            return HttpResponse('Invalid city selected', status=400)

        if input_category_dtree not in category_choices:
            return HttpResponse('Invalid category selected', status=400)

        input_date_dtree_str = request.POST.get('date')

        try:
            prediction = Prediction.objects.create(date=input_date_dtree_str)  # Create a new Prediction object
        except ValueError:
            return HttpResponse('Invalid date format or other error occurred', status=400)

        input_date_dtree = datetime.strptime(input_date_dtree_str, '%Y-%m-%d').date()
        input_year = input_date_dtree.year
        input_month = input_date_dtree.month
        input_day = input_date_dtree.day

        # Define the path to the model based on the selected category
        model_path = f'home/static/City_{input_city_dtree}_Category_{input_category_dtree}_demand_model.pkl'

        try:
            dtree_demand_model = joblib.load(model_path)
        except FileNotFoundError:
            return HttpResponse('Model not found for the selected category', status=400)

        input_features = [[input_year, input_month, input_day]]
        demand_prediction_dtree = dtree_demand_model.predict(input_features)

        # Similar modifications for supply model path
        supply_model_path = f'home/static/City_{input_city_dtree}_Category_{input_category_dtree}_supply_model.pkl'
        try:
            dtree_supply_model = joblib.load(supply_model_path)
        except FileNotFoundError:
            return HttpResponse('Supply Model not found for the selected category', status=400)

        supply_prediction_dtree = dtree_supply_model.predict(input_features)

        Demand_Graph = "static/image/Demand_plot.jpg"
        Supply_Graph = "static/image/Supply_plot.jpg"
        Demand_And_Supply_Graph = "static/image/DemandAndSupply_plot.jpg"
        Dtree_Scatter_Graph = "static/image/dtree_scatter_plot.png"

        prediction_result3 = PredictionResult3(
            input_city = input_city_dtree,
            input_category=input_category_dtree,
            input_date=input_date_dtree,
            demand_prediction=demand_prediction_dtree[0],
            supply_prediction=supply_prediction_dtree[0],
            Demand_Graph=Demand_Graph,
            Supply_Graph=Supply_Graph,
            Demand_And_Supply_Graph=Demand_And_Supply_Graph,
            Dtree_Scatter_Graph=Dtree_Scatter_Graph,
        )
        prediction_result3.save()

        context = {
            'input_city': input_city_dtree,
            'input_category': input_category_dtree,
            'input_date': input_date_dtree,
            'demand_prediction': demand_prediction_dtree[0],
            'supply_prediction': supply_prediction_dtree[0],
            'Demand_Graph': Demand_Graph,
            'Supply_Graph': Supply_Graph,
            'Demand_And_Supply_Graph': Demand_And_Supply_Graph,
            'Dtree_Scatter_Graph': Dtree_Scatter_Graph,
        }

        return render(request, 'prediction_results3.html', context)
    else:
        return render(request, 'prediction_results3.html')


def predictResult3(request):
    return render(request, 'prediction_results3.html')


