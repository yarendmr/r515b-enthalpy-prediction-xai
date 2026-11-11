"""Fast checks for input integrity, metrics, and estimator configuration."""
import unittest

from r515b.data import FILES, check_data
from r515b.modeling import genel_skor, metrik_hesapla, model_kur
from sklearn.ensemble import ExtraTreesRegressor


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = metrik_hesapla([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(result["R2"], 1.0)
        for key in ("RMSE", "MAE", "Tail_MAE", "Maks_Hata"):
            self.assertEqual(result[key], 0)

    def test_original_composite_score_formula(self):
        result = genel_skor({"R2": 0.9, "RMSE": 0.5, "MAE": 0.2})
        self.assertAlmostEqual(result, 0.84)

    def test_extra_trees_parameters(self):
        model = model_kur("ET", {
            "n_estimators": 250, "max_depth": None,
            "min_samples_leaf": 1, "max_features": "sqrt"
        })
        self.assertIsInstance(model, ExtraTreesRegressor)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.n_estimators, 250)


class InputDataTests(unittest.TestCase):
    @unittest.skipUnless(all(p.is_file() for p, _, _ in FILES.values()),
                         "Source workbooks are not tracked in Git")
    def test_data_format(self):
        self.assertEqual(set(check_data()), {
            ("Superheated vapor", 578, 5),
            ("Saturated properties", 136, 8),
        })


if __name__ == "__main__":
    unittest.main()
