"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")

application = get_wsgi_application()


# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.xgboost_classifier_1.Xgboost_1 import XgbClassifier

try:
    registry = MLRegistry() # create ML registry
    # xgboost_classifier
    rf = XgbClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="xgboost_classifier_1",
                            algorithm_object=rf,
                            algorithm_name="xgboost_classifier",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="ISA",
                            algorithm_description="xgboost_classifier for current transformers",
                            algorithm_code=inspect.getsource(XgbClassifier))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))