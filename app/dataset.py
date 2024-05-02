from flask_restx import Resource, Namespace,reqparse,fields
import os,json
from .extensions import api
from werkzeug.datastructures import FileStorage
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd


# load=Namespace("Dataset APIs")

data=pd.read_csv(os.getenv("DataSetPath"))
target=None
train_data,test_data=None,None
dataPrepared=None
problem,mlModel,bot=os.getenv("MlProblemType"),None,None
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# upload_parser = reqparse.RequestParser()
# upload_parser.add_argument('file', type=FileStorage, location='files', required=True, help='File to upload')

# def loadData(file_path):
#     global data
#     data=pd.read_csv(file_path)
#     d=data.head()
#     listdata=[]
#     for i in range(0,6):
#         dic={}
#         for j in d:
#             dic[j]=data.iloc[i][j]
#         listdata.append(dic)
#     return listdata



# @load.route('/upload')
# class FileUpload(Resource): 
#     @api.expect(upload_parser)
#     def post(self):
#         # Parse the incoming request
#         args = upload_parser.parse_args()
#         file = args['file']
#         # Check if the file has a valid filename
#         if file.filename == '':
#             return {'error': 'No selected file'}, 400
#         # Save the uploaded file
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(file_path)
#         head=loadData(file_path)
#         return {'message': f'File {file.filename} uploaded successfully..!.',
#                 "DataSetshape":str(data.shape),
#                 "head()":head
#                 }, 200

# ----------------------------preparation---------------------------------
preparation=Namespace("Data Preparation APIs")


@preparation.route('/1.checkForMissingValues')
class IsNull(Resource):
    def get(self):
        columns=data.columns
        missingValues=data.isnull().sum()
        missingValuedict={}
        for i in range(len(columns)):
            missingValuedict[columns[i]]=str(missingValues.iloc[i])
        return {"Null Values": str(data.isnull().values.any()),
                "missingvalues":missingValuedict,
                "Total Missing Values":str(data.isnull().sum().sum())
                }, 200

missingValueParcer=reqparse.RequestParser()
missingValueParcer.add_argument('Action',type=str,
                                choices=
                                ['remove entire row', 
                                 'remove entire column', 
                                 'replace missing values with mean',
                                 'replace missing values with median'],
                                 required=True,help="handling Missing Value"    
)
@preparation.route('/2.handleMissingValues')
class handleMissingValues(Resource):
    @api.expect(missingValueParcer)
    def get(self):
        args = missingValueParcer.parse_args()  # Parse the request
        selected_action = args['Action']  # Get the chosen value
        columns=data.columns
        missingValues=list(data.isnull().sum())
        missingValueColumns=[]
        for i in range(len(columns)):
            if missingValues[i]>0:
                missingValueColumns.append(columns[i])
        for i in missingValueColumns:
            match selected_action:
                case "remove entire row":
                    data.dropna(subset=[i])
                case "remove entire column":
                    data.drop(i,axis=1,inplace=True)
                case "replace missing values with median":
                    median=data[i].median()
                    data[i].fillna(median,inplace=True)
                case "replace missing values with mean":
                    mean=data[i].mean()
                    data[i].fillna(mean,inplace=True)
        return {'selected_Action': selected_action,
                "Action Status":"Successful."
                }, 200

CategoricalParcer=reqparse.RequestParser()
CategoricalParcer.add_argument('Encoder',type=str,choices=["Ordinal Encoder", "one-hot encoding"],
                                 required=True,help="handling categorical Value"    
)
@preparation.route('/3.checkForCategoricalData')
class CategoricalDataCheck(Resource):
    def get(self):
        categoricalStatus="O" in list(data.dtypes)
        return{"categoricalStatus":str(categoricalStatus)},200

def updateDF(dataFrame):
    global data
    data=dataFrame

@preparation.route('/4.handleCategoricalData')
class HandleCategoricalData(Resource):
    @api.expect(CategoricalParcer)
    def get(self):
        args = CategoricalParcer.parse_args()  # Parse the request
        selected_encoder = args['Encoder']
        columns=data.columns
        dtypes=data.dtypes
        catColumns=[]
        for i in range(len(columns)):
            if(dtypes[i]=='O'):
                catColumns.append(columns[i])
        match selected_encoder:
            case "Ordinal Encoder":
                ordinal_encoder=OrdinalEncoder()
                data_cat=data[catColumns]
                data[catColumns]=ordinal_encoder.fit_transform(data_cat)
                return {
                    "Selected Encoder":selected_encoder,
                    "Status":"encoding Done successful."
                    },200
            case "one-hot encoding":
                one_hot_encoder=OneHotEncoder(sparse_output=False)
                data_cat=data[catColumns]
                data_cat_1hot=one_hot_encoder.fit_transform(data_cat)
                one_hot_df = pd.DataFrame(data_cat_1hot, columns=one_hot_encoder.get_feature_names_out(catColumns))
                dataFrame = pd.concat([data, one_hot_df], axis=1)
                dataFrame.drop(catColumns,axis=1,inplace=True)
                updateDF(dataFrame)

                return {
                    "Selected Encoder":selected_encoder,
                    "Status":"encoding Done successful."
                },200
# -----------------------------Corelation---------------------------
feature=Namespace("Feature APIs")
def setTarget(val):
    global target
    target=val
targetParcer=reqparse.RequestParser()
targetParcer.add_argument("Target",type=str,choices=list(data.columns),required=True,help="Select Target Value")
@feature.route("/1.selectTarget")
class SelectTarget(Resource):
    @api.expect(targetParcer)
    def get(self):
        args = targetParcer.parse_args()  # Parse the request
        selected_encoder = args['Target']
        setTarget(selected_encoder)
        return {"selected Target":target},200

@feature.route("/2.getCorrelation")
class GetCorrelation(Resource):
    def get(self):
        corr_matrix=data.corr()
        return {"message":"correlation of features with respect to target in descending order.",
                "correlations":str(corr_matrix[target].sort_values(ascending=False)).split("\n")[:-1]
                },200

feature_select_parser = reqparse.RequestParser()
feature_select_parser.add_argument('fearure', type=str, action='append', 
                                 choices=list(data.columns),
                                 required=True, help='Features to select:'+str(list(data.columns)))
def updateDataset(features):
    global dataPrepared
    dataPrepared=data[features]
@feature.route("/3.selectFeaturesToTrain")
class SelectFeaturesToTrain(Resource):
    @api.expect(feature_select_parser)
    def post(self):
        args = feature_select_parser.parse_args()  # Parse the query parameters
        selected_features = args['fearure']
        updateDataset(selected_features)
        return {"selected Features":selected_features
                },200

scalerParcer=reqparse.RequestParser()
scalerParcer.add_argument("scaler",type=str,choices=['MinMaxScaler',"StandardScaler"],required=True,help="Select type of Scaling")
def updatePreparedData(dset):
    global dataPrepared
    dataPrepared=dset
@feature.route('/4.DataScaling')
class performDataScaling(Resource):
    @api.expect(scalerParcer)
    def get(self):
        args = scalerParcer.parse_args()  # Parse the query parameters
        selected_scaler = args['scaler']
        match selected_scaler:
            case "MinMaxScaler":
                scaler=MinMaxScaler()
                updatePreparedData(scaler.fit_transform(dataPrepared))
            case "StandardScaler":
                scaler=StandardScaler()
                updatePreparedData(scaler.fit_transform(dataPrepared))
        return {"Message":f"Data scaled using {selected_scaler} Successfully."},200

# -----------------------------Model APIs-----------------------------


model=Namespace("Model APIs")

split_parser = reqparse.RequestParser()
split_parser.add_argument('split', type=float, required=True, help='Test size for TestTrainsplit [like 10,20,30]')

def updateTestTrain(train,test):
    global train_data
    train_data=train
    global test_data
    test_data=test

@model.route("/1.TestTrainSplit")
class TestTrainSplit(Resource):
   @api.expect(split_parser)
   def post(self):
       arg=split_parser.parse_args()
       split=arg["split"]
       concatTarget=pd.concat([pd.DataFrame(dataPrepared), pd.DataFrame(data[target])], axis=1)
       updatePreparedData(concatTarget)
       train_set,test_set=train_test_split(dataPrepared,test_size=split/100,random_state=42)
       updateTestTrain(train_set,test_set)
       return {"Train_size":str(train_data.shape),"Test_size":str(test_data.shape)},200
# problem_parcer=reqparse.RequestParser()
# problem_parcer.add_argument("ProblemType",type='str',choices=["Regression","Classification"],required=True,help="Select type of ML Problem")
# def setProblem(prob):
#     global problem
#     problem=prob
# @model.route("/2.TypeMlProblem")
# class MLProblemType(Resource):
#     @api.expect(problem_parcer)
#     def get(self):
#         args=problem_parcer.parse_args()
#         typeSelected=args['ProblemType']
#         setProblem(typeSelected)
#         return {"Ml Problem selected":problem},200

mlModels={
    "Regression":['LinearRegression','DecisionTreeRegressor','RandomForestRegressor'],
    "Classification":["SGDClassifier",'SVC']
}

model_parcer=reqparse.RequestParser()
model_parcer.add_argument('model',type=str,choices=mlModels[problem],required=True,help="Select type of ML Model")

def setMlModel(model_ml):
    global mlModel
    mlModel=model_ml

@model.route("/2.ModelSelection")
class SelectModel(Resource):
    @api.expect(model_parcer)
    def get(self):
        args=model_parcer.parse_args()
        modelSelected=args['model']
        setMlModel(modelSelected)
        return {"message":f'model selected {mlModel}'},200
def setBot(bb):
    global bot
    bot=bb

from sklearn.linear_model import LinearRegression,SGDClassifier
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error,precision_score,recall_score,f1_score
import numpy as np

@model.route("/3.train")
class TrainModel(Resource):
    def get(self):
        train_x=train_data.drop(target,axis=1)
        train_y=train_data[target].copy()
        match mlModel:
            case "LinearRegression":
                lin_reg=LinearRegression()
                lin_reg.fit(train_x,train_y)
                setBot(lin_reg)
            case "DecisionTreeRegressor":
                tree_reg=DecisionTreeRegressor(random_state=42)
                tree_reg.fit(train_x,train_y)
                setBot(tree_reg)
            case "RandomForestRegressor":
                forest_reg=RandomForestRegressor()
                forest_reg.fit(train_x,train_y)
                setBot(forest_reg)
            case "SGDClassifier":
                sgd_clf=SGDClassifier(random_state=42)
                sgd_clf.fit(train_x,train_y)
                setBot(sgd_clf)
            case "SVC":
                svm_clf=SVC()
                svm_clf.fit(train_x,train_y)
                setBot(svm_clf)
        return {"Message":'Model Trained Successfully..!'},200

@model.route("/4.Evaluate")
class ModelEcaluate(Resource):
    def get(self):
        res={}
        test_x=test_data.drop(target,axis=1)
        test_y=test_data[target].copy()
        train_x=train_data.drop(target,axis=1)
        train_y=train_data[target].copy()
        match problem:
            case "Regression":
                predictions=bot.predict(train_x)
                mse=mean_squared_error(train_y,predictions)
                rmse=np.sqrt(mse)
                res["RMSE on Train data"]=str(rmse)
                predictions=bot.predict(test_x)
                mse=mean_squared_error(test_y,predictions)
                rmse=np.sqrt(mse)
                res["RMSE on Test data"]=str(rmse)
            case "Classification":
                predictions=bot.predict(train_x)
                res["train"]['precision_score']=str( precision_score(train_y,predictions))
                res["train"]['recall_score']=str(recall_score(train_y,predictions))
                res["train"]['f1_score']=str(f1_score(train_y,predictions))
                predictions=bot.predict(test_x)
                res["test"]['precision_score']=str( precision_score(test_y,predictions))
                res["test"]['recall_score']=str(recall_score(test_y,predictions))
                res["test"]['f1_score']=str(f1_score(test_y,predictions))
        return {"Model Performance":res},200


