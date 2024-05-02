from flask import Flask
from .extensions import api
from.dataset import preparation,feature,model


def create_app():
    app=Flask(__name__)
    api.init_app(app,title='MLaaS',description='Run the APIs in order..!    ')
    # api.add_namespace(load)
    api.add_namespace(preparation)
    api.add_namespace(feature)
    api.add_namespace(model)
    return app
