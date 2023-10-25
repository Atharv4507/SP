from django.test import TestCase
from .models import PredictionResult, PredictionResult2, PredictionResult3


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
