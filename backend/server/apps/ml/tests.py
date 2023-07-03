# add at the beginning of the file:
import inspect
from apps.ml.registry import MLRegistry
from django.test import TestCase
from apps.ml.xgboost_classifier_1.Xgboost_1 import XgbClassifier
# ...
# the rest of the code
# ...

# add below method to MLTests class:







class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
            "capacitancia_total": 0.03,
            "factor_de_potencia_10kv_ctotal": 1,
            "factor_de_potencia_2.5kv_ctotal": -1,
            "medida_resistencia_aislamiento_nucleo_1_ultimo": 4041,
            "medida_resistencia_aislamiento_nucleo_2_ultimo": 4545,
            "medida_resistencia_aislamiento_nucleo_3_ultimo": 2000,
            "medida_resistencia_aislamiento_nucleo_4_ultimo": 1000,
            "medida_resistencia_aislamiento_nucleo_5_ultimo": 10000,
            "medida_resistencia_aislamiento_nucleo_6_ultimo":-8691, "severidad_por_termografia":5,
            "Fabricante": 3,
            "CHANGED_estado_cajetin": 5,
            "CHANGED_estado_porcelana": 5,
            "CHANGED_fuga_de_aceite": 5,
            "CHANGED_inspeccion_diafragma": 5,
            "CHANGED_inspeccion_visual_general": 5 ,
            "CHANGED_nivel_de_aceite":5
        }
        my_alg = XgbClassifier()
        response = my_alg.compute_prediction(input_data)
        print(response)
    def test_registry(self):
            registry = MLRegistry()
            self.assertEqual(len(registry.endpoints), 0)
            endpoint_name = "income_classifier"
            algorithm_object = XgbClassifier()
            algorithm_name = "XGboost"
            algorithm_status = "production"
            algorithm_version = "0.0.1"
            algorithm_owner = "Piotr"
            algorithm_description = "Random Forest with simple pre- and post-processing"
            algorithm_code = inspect.getsource(XgbClassifier)
            # add to registry
            registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
            self.assertEqual(len(registry.endpoints), 1)
        
