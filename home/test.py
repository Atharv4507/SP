from django.test import TestCase
from .models import Prediction, PredictionResult, PredictionResult2, PredictionResult3
from django.contrib.auth.models import User
from django.urls import reverse
from datetime import date
import joblib


class SignLoginLogoutTest(TestCase):
    def test_sign_view(self):
        response = self.client.post(reverse('sign'), {
            'username': 'testuser',
            'email': 'test@example.com',
            'password1': 'password123',
            'password2': 'password123',
        })
        self.assertEqual(response.status_code, 302)  # Expect a redirect
        self.assertRedirects(response, reverse('loginp'))

    def test_sign_view_password_mismatch(self):
        response = self.client.post(reverse('sign'), {
            'username': 'testuser',
            'email': 'test@example.com',
            'password1': 'password123',
            'password2': 'password456',  # Mismatched password
        })
        self.assertEqual(response.status_code, 200)  # Expect to stay on the sign page
        self.assertContains(response, "your passwords doesn't matching")

    def test_login_view(self):
        user = User.objects.create_user(username='testuser', password='password123')
        response = self.client.post(reverse('loginp'), {
            'username': 'testuser',
            'pass': 'password123',
        })
        self.assertEqual(response.status_code, 302)  # Expect a redirect
        self.assertRedirects(response, reverse('home'))

    def test_login_view_invalid_credentials(self):
        response = self.client.post(reverse('loginp'), {
            'username': 'nonexistentuser',
            'pass': 'invalidpassword',
        })
        self.assertEqual(response.status_code, 200)  # Expect to stay on the login page
        self.assertContains(response, "Username or password is incorrect")

    def test_logout_view(self):
        user = User.objects.create_user(username='testuser', password='password123')
        self.client.force_login(user)
        response = self.client.get(reverse('logoutp'))
        self.assertEqual(response.status_code, 302)  # Expect a redirect
        self.assertRedirects(response, reverse('loginp'))


class PredictionResultTestCase(TestCase):
    def test_str_method(self):
        result = PredictionResult(input_date="2023-10-17", demand_prediction=10.0, supply_prediction=5.0)
        self.assertEqual(str(result), "2023-10-17")


class PredictionResult2TestCase(TestCase):
    def test_str_method(self):
        result = PredictionResult2(input_category="Health and beauty", input_date="2023-10-17", demand_prediction=10.0, supply_prediction=5.0)
        self.assertEqual(str(result), "Health and beauty")


class PredictionResult3TestCase(TestCase):
    def test_str_method(self):
        result = PredictionResult3(input_city="Bangalore", input_category="Health and beauty", input_date="2023-10-17", demand_prediction=10.0, supply_prediction=5.0)
        self.assertEqual(str(result), "Bangalore")

    def test_create_result(self):
        result = PredictionResult3(input_city="Bangalore", input_category="Health and beauty", input_date="2023-10-17", demand_prediction=10.0, supply_prediction=5.0)
        result.save()
        self.assertEqual(PredictionResult3.objects.count(), 1)


# class PredictDemandSupplyDtreeTest(TestCase):
#
#     def test_predict_demand_supply_dtree_valid(self):
#         input_date = '2023-10-25'
#         response = self.client.post(reverse('predict_demand_supply_dtree'), {'date': input_date})
#
#         # Check the response status code
#         self.assertEqual(response.status_code, 200)
#
#         # Check that a Prediction object was created
#         self.assertEqual(Prediction.objects.count(), 1)
#
#         # Check that a PredictionResult object was created
#         self.assertEqual(PredictionResult.objects.count(), 1)
#
#         # Check the context in the response
#         context = response.context
#         self.assertEqual(context['input_date'], date(2023, 10, 25))
#
#     def test_predict_demand_supply_dtree_invalid_date(self):
#         input_date = 'invalid_date'
#         response = self.client.post(reverse('predict_demand_supply_dtree'), {'date': input_date})
#
#         # Check the response status code for invalid input
#         self.assertEqual(response.status_code, 400)
#
#         # Check that no Prediction object was created
#         self.assertEqual(Prediction.objects.count(), 0)
#
#         # Check that no PredictionResult object was created
#         self.assertEqual(PredictionResult.objects.count(), 0)
#
#     def test_predict_demand_supply_dtree_get(self):
#         response = self.client.get(reverse('predict_demand_supply_dtree'))
#
#         # Check the response status code for a GET request
#         self.assertEqual(response.status_code, 200)
#
#     def test_predict_result(self):
#         response = self.client.get(reverse('predict_result'))
#
#         # Check the response status code
#         self.assertEqual(response.status_code, 200)
#
#         # You can add more assertions to check the content of the response if needed
#
#     def setUp(self):
#         # Load models and save them as class attributes for reuse in tests
#         self.dtree_demand_model = joblib.load('home/static/ml_dtree_demand_model.pkl')
#         self.dtree_supply_model = joblib.load('home/static/ml_dtree_supply_model.pkl')
#
#     def tearDown(self):
#         # Clean up any objects created during the tests if needed
#         Prediction.objects.all().delete()
#         PredictionResult.objects.all().delete()
#
#
# class PCACWTest(TestCase):
#
#     def test_pcacw_valid(self):
#         # Prepare valid input data
#         input_city = 'Delhi'
#         input_category = 'Health and beauty'
#         input_date = '2023-10-25'
#
#         # Create the corresponding model file paths
#         demand_model_path = f'home/static/City_{input_city}_Category_{input_category}_demand_model.pkl'
#         supply_model_path = f'home/static/City_{input_city}_Category_{input_category}_supply_model.pkl'
#
#         # Create PredictionResult3 objects based on the model paths
#         prediction_result3 = PredictionResult3(
#             input_city=input_city,
#             input_category=input_category,
#             input_date=date(2023, 10, 25),
#             demand_prediction=10.0,  # Replace with your expected value
#             supply_prediction=5.0,  # Replace with your expected value
#             Demand_Graph="static/image/Demand_plot.jpg",
#             Supply_Graph="static/image/Supply_plot.jpg",
#             Demand_And_Supply_Graph="static/image/DemandAndSupply_plot.jpg",
#             Dtree_Scatter_Graph="static/image/dtree_scatter_plot.png",
#         )
#         prediction_result3.save()
#
#         response = self.client.post(reverse('pcacw'), {
#             'citySelect': input_city,
#             'categorySelect': input_category,
#             'date': input_date,
#         })
#
#         # Check the response status code
#         self.assertEqual(response.status_code, 200)
#
#         # Check that a Prediction object was created
#         self.assertEqual(Prediction.objects.count(), 1)
#
#         # Check that a PredictionResult3 object was created
#         self.assertEqual(PredictionResult3.objects.count(), 1)
#
#         # Check the context in the response
#         context = response.context
#         self.assertEqual(context['input_city'], input_city)
#         self.assertEqual(context['input_category'], input_category)
#         self.assertEqual(context['input_date'], date(2023, 10, 25))
#
#     def test_pcacw_invalid_city(self):
#         response = self.client.post(reverse('pcacw'), {
#             'citySelect': 'InvalidCity',
#             'categorySelect': 'Health and beauty',
#             'date': '2023-10-25',
#         })
#
#         # Check the response status code for an invalid city
#         self.assertEqual(response.status_code, 400)
#
#         # Check that no Prediction object was created
#         self.assertEqual(Prediction.objects.count(), 0)
#
#         # Check that no PredictionResult3 object was created
#         self.assertEqual(PredictionResult3.objects.count(), 0)
#
#     def test_pcacw_invalid_category(self):
#         response = self.client.post(reverse('pcacw'), {
#             'citySelect': 'Delhi',
#             'categorySelect': 'InvalidCategory',
#             'date': '2023-10-25',
#         })
#
#         # Check the response status code for an invalid category
#         self.assertEqual(response.status_code, 400)
#
#         # Check that no Prediction object was created
#         self.assertEqual(Prediction.objects.count(), 0)
#
#         # Check that no PredictionResult3 object was created
#         self.assertEqual(PredictionResult3.objects.count(), 0)
#
#     def test_pcacw_invalid_date(self):
#         response = self.client.post(reverse('pcacw'), {
#             'citySelect': 'Delhi',
#             'categorySelect': 'Health and beauty',
#             'date': 'invalid_date',
#         })
#
#         # Check the response status code for an invalid date
#         self.assertEqual(response.status_code, 400)
#
#         # Check that no Prediction object was created
#         self.assertEqual(Prediction.objects.count(), 0)
#
#         # Check that no PredictionResult3 object was created
#         self.assertEqual(PredictionResult3.objects.count(), 0)
#
#     def test_pcacw_get(self):
#         response = self.client.get(reverse('pcacw'))
#
#         # Check the response status code for a GET request
#         self.assertEqual(response.status_code, 200)
#
#     def test_predict_result3(self):
#         response = self.client.get(reverse('predictResult3'))
#
#         # Check the response status code
#         self.assertEqual(response.status_code, 200)
#
#

