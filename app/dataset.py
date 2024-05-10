from flask_restx import Resource, Namespace,reqparse
from flask import send_file, abort,url_for,render_template,make_response
import os,shutil
from .extensions import api
from werkzeug.datastructures import FileStorage
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import requests
import matplotlib.pyplot as plt


load=Namespace("Dataset APIs",description="load data for training")

data=None
# data=pd.read_csv(os.getenv("DataSetPath"))
target=None
train_data,test_data=None,None
dataPrepared=None
problem,mlModel,bot=None,None,None
# problem,mlModel,bot=os.getenv("MlProblemType"),None,None
UPLOAD_FOLDER = 'uploads'

upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', type=FileStorage, location='files', required=True, help='File to upload')

def loadData(file_path):
    global data
    data=pd.read_csv(file_path)
    d=data.head()
    listdata=[]
    for i in range(0,6):
        dic={}
        for j in d:
            dic[j]=data.iloc[i][j]
        listdata.append(dic)
    return listdata



@load.route('/upload')
class FileUpload(Resource): 
    @api.expect(upload_parser)
    def post(self):
        # Parse the incoming request
        args = upload_parser.parse_args()
        file = args['file']
        # Check if the file has a valid filename
        if file.filename == '':
            return {'error': 'No selected file'}, 400
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        head=loadData(file_path)
        return {'message': f'File {file.filename} uploaded successfully..!.',
                "DataSetshape":str(data.shape),
                "head()":head
                }, 200

# ----------------------------preparation---------------------------------
preparation=Namespace("Data Preparation APIs",description="preprocess the data for Model Training")


@preparation.route('/1.checkForMissingValues')
class IsNull(Resource):
    def get(self):
        if data is None:
             return {"Error":"upload the dataset first and try again."},400
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
        if data is None:
             return {"Error":"upload the dataset first and try again."},400
        columns=data.columns
        missingValues=list(data.isnull().sum())
        missingValueColumns=[]
        for i in range(len(columns)):
            if missingValues[i]>0:
                missingValueColumns.append(columns[i])
        for i in missingValueColumns:
            if(selected_action=="remove entire row"):
                    data.dropna(subset=[i])
            elif selected_action=="remove entire column":
                    data.drop(i,axis=1,inplace=True)
            elif selected_action== "replace missing values with median":
                    median=data[i].median()
                    data[i].fillna(median,inplace=True)
            elif selected_action== "replace missing values with mean":
                    mean=data[i].mean()
                    data[i].fillna(mean,inplace=True)
        return {'selected_Action': selected_action,
                "Action Status":"Successful.",
                "updated Columns":missingValueColumns
                }, 200

CategoricalParcer=reqparse.RequestParser()
CategoricalParcer.add_argument('Encoder',type=str,choices=["Ordinal Encoder", "one-hot encoding"],
                                 required=True,help="handling categorical Value"    
)
@preparation.route('/3.checkForCategoricalData')
class CategoricalDataCheck(Resource):
    def get(self):
        if data is None:
             return {"Error":"upload the dataset first and try again."},400
        cat_columns=[]
        categoricalStatus=None
        columns=data.columns
        dtypes=data.dtypes
        for i in range(len(columns)):
            if(dtypes[i]=='O'):
                categoricalStatus=True
                cat_columns.append(columns[i])
        return{"categoricalStatus":str(categoricalStatus),"categorial columns":cat_columns},200

def updateDF(dataFrame):
    global data
    data=dataFrame

@preparation.route('/4.handleCategoricalData')
class HandleCategoricalData(Resource):
    @api.expect(CategoricalParcer)
    def get(self):
        args = CategoricalParcer.parse_args()  # Parse the request
        selected_encoder = args['Encoder']
        if data is None:
             return {"Error":"upload the dataset first and try again."},400
        columns=data.columns
        dtypes=data.dtypes
        catColumns=[]
        for i in range(len(columns)):
            if(dtypes[i]=='O'):
                catColumns.append(columns[i])
        if selected_encoder=="Ordinal Encoder":
                ordinal_encoder=OrdinalEncoder()
                data_cat=data[catColumns]
                data[catColumns]=ordinal_encoder.fit_transform(data_cat)
                return {
                    "Selected Encoder":selected_encoder,
                    "Status":"encoding Done successful.",
                    "Columns Encoded":catColumns
                    },200
        if selected_encoder=="one-hot encoding":
                one_hot_encoder=OneHotEncoder(sparse_output=False)
                data_cat=data[catColumns]
                data_cat_1hot=one_hot_encoder.fit_transform(data_cat)
                one_hot_df = pd.DataFrame(data_cat_1hot, columns=one_hot_encoder.get_feature_names_out(catColumns))
                dataFrame = pd.concat([data, one_hot_df], axis=1)
                dataFrame.drop(catColumns,axis=1,inplace=True)
                updateDF(dataFrame)

                return {
                    "Selected Encoder":selected_encoder,
                    "Status":"encoding Done successful.",
                    "Columns Encoded":list(one_hot_encoder.get_feature_names_out())
                },200
# -----------------------------Corelation---------------------------
feature=Namespace("Feature APIs",description="feature selection operation")
def setTarget(val):
    global target
    target=val
targetParcer=reqparse.RequestParser()
targetParcer.add_argument("Target",type=str,required=True,help="Provide Target Feature")
@feature.route("/1.selectTarget")
class SelectTarget(Resource):
    @api.expect(targetParcer)
    def get(self):
        if data is None:
             return {"Error":"upload the dataset first and try again."},400
        args = targetParcer.parse_args()  # Parse the request
        selected_Target = args['Target']
        if ((selected_Target=='')or (selected_Target not in list(data.columns))or (selected_Target==None)):
             return{"Error":"Invalid Target Specified","Available Targets":list(data.columns)},400
        setTarget(selected_Target)
        return {"selected Target":target},200

@feature.route("/2.getCorrelation")
class GetCorrelation(Resource):
    def get(self):
        if data is None:
             return {"Error":"upload the dataset first and try again."},400
        if target==None:
          return {"error":"Select Target and Try again."},400
        corr_matrix=data.corr()
        return {"message":"correlation of features with respect to target in descending order.",
                "correlations":str(corr_matrix[target].sort_values(ascending=False)).split("\n")[:-1]
                },200

feature_select_parser = reqparse.RequestParser()
feature_select_parser.add_argument('fearure', type=str, action='append',
                                 required=True, help='Features to train the model')
def updateDataset(features):
    global dataPrepared
    dataPrepared=data[features]
@feature.route("/3.selectFeaturesToTrain")
class SelectFeaturesToTrain(Resource):
    @api.expect(feature_select_parser)
    def post(self):
        if data is None:
             return {"Error":"upload the dataset first and try again."},400
        args = feature_select_parser.parse_args()  # Parse the query parameters
        selected_features = args['fearure']
        for i in selected_features:
             if i not in list(data.columns):
                  return {"Error":"Invalid Features Provided. Check the spelling and try again.","Available Features":list(data.columns)},400
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
        if dataPrepared is None:
             return {"Error":"Select Features from the dataset first and try again."},400
        args = scalerParcer.parse_args()  # Parse the query parameters
        selected_scaler = args['scaler']
        if selected_scaler=="MinMaxScaler":
                scaler=MinMaxScaler()
                updatePreparedData(scaler.fit_transform(dataPrepared))
        elif selected_scaler=="StandardScaler":
                scaler=StandardScaler()
                updatePreparedData(scaler.fit_transform(dataPrepared))
        return {"Message":f"Data scaled using {selected_scaler} Successfully."},200

# -----------------------------Model APIs-----------------------------


model=Namespace("Model_APIs",description="Start training the ML models")

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
       if data is None:
             return {"Error":"upload the dataset first and try again."},400
       if dataPrepared is None:
             return {"Error":"Select Features from the dataset first and try again."},400
       arg=split_parser.parse_args()
       split=arg["split"]
       concatTarget=pd.concat([pd.DataFrame(dataPrepared), pd.DataFrame(data[target])], axis=1)
       updatePreparedData(concatTarget)
       train_set,test_set=train_test_split(dataPrepared,test_size=split/100,random_state=42)
       updateTestTrain(train_set,test_set)
       return {"Train_size":str(train_data.shape),"Test_size":str(test_data.shape)},200
problem_parcer=reqparse.RequestParser()
problem_parcer.add_argument("ProblemType",type=str,choices=["Regression","Classification"],required=True,help="Select type of ML Problem")
def setProblem(prob):
    global problem
    problem=prob
@model.route("/2.TypeMlProblem")
class MLProblemType(Resource):
    @api.expect(problem_parcer)
    def get(self):
        args=problem_parcer.parse_args()
        typeSelected=args['ProblemType']
        setProblem(typeSelected)
        return {"Ml Problem selected":problem},200

mlModels=['LinearRegression','DecisionTreeRegressor','RandomForestRegressor',"SGDClassifier",'SVC']
    
# mlModels={
#     "Regression":['LinearRegression','DecisionTreeRegressor','RandomForestRegressor'],
#     "Classification":["SGDClassifier",'SVC']
# }

model_parcer=reqparse.RequestParser()
model_parcer.add_argument('model',type=str,
                          choices=['LinearRegression','DecisionTreeRegressor','RandomForestRegressor',"SGDClassifier",'SVC'],
                          required=True,help="Select type of ML Model")

def setMlModel(model_ml):
    global mlModel
    mlModel=model_ml

@model.route("/3.ModelSelection")
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

@model.route("/4.train")
class TrainModel(Resource):
    def get(self):
        if train_data is None or test_data is None:
             return {"Error":"Split the data into train and test Dataset first and try again."},400
        if mlModel==None:
             return {"Error":"Select Model To train first and try again."},400
        if target==None:
          return {"error":"Select Target and Try again."},400
        train_x=train_data.drop(target,axis=1)
        train_y=train_data[target].copy()
        if mlModel=="LinearRegression":
                lin_reg=LinearRegression()
                lin_reg.fit(train_x,train_y)
                setBot(lin_reg)
        elif mlModel=="DecisionTreeRegressor":
                tree_reg=DecisionTreeRegressor(random_state=42)
                tree_reg.fit(train_x,train_y)
                setBot(tree_reg)
        if mlModel== "RandomForestRegressor":
                forest_reg=RandomForestRegressor()
                forest_reg.fit(train_x,train_y)
                setBot(forest_reg)
        if mlModel== "SGDClassifier":
                sgd_clf=SGDClassifier(random_state=42)
                sgd_clf.fit(train_x,train_y)
                setBot(sgd_clf)
        if mlModel== "SVC":
                svm_clf=SVC()
                svm_clf.fit(train_x,train_y)
                setBot(svm_clf)
        joblib.dump(bot, 'app/TrainedModel.joblib')
        return {"Message":'Model Trained Successfully..!'},200

@model.route("/5.Evaluate")
class ModelEcaluate(Resource):
    def get(self):
        res={}
        if train_data is None or test_data is None:
             return {"Error":"Split the data into train and test Dataset first and try again."},400
        if target==None:
          return {"error":"Select Target and Try again."},400
        if problem==None:
             return {"Error":"Select the type of ML problem first and try again."},400
        if bot==None:
             return {"Error":"Train the Model first and try again."},400
        test_x=test_data.drop(target,axis=1)
        test_y=test_data[target].copy()
        train_x=train_data.drop(target,axis=1)
        train_y=train_data[target].copy()
        if problem== "Regression":
                predictions=bot.predict(train_x)
                mse=mean_squared_error(train_y,predictions)
                rmse=np.sqrt(mse)
                res["RMSE on Train data"]=str(rmse)
                predictions=bot.predict(test_x)
                mse=mean_squared_error(test_y,predictions)
                rmse=np.sqrt(mse)
                res["RMSE on Test data"]=str(rmse)
        elif problem== "Classification":
                predictions=bot.predict(train_x)
                res["train"]['precision_score']=str( precision_score(train_y,predictions))
                res["train"]['recall_score']=str(recall_score(train_y,predictions))
                res["train"]['f1_score']=str(f1_score(train_y,predictions))
                predictions=bot.predict(test_x)
                res["test"]['precision_score']=str( precision_score(test_y,predictions))
                res["test"]['recall_score']=str(recall_score(test_y,predictions))
                res["test"]['f1_score']=str(f1_score(test_y,predictions))
        return {"Model Performance":res},200

hidden=Namespace('hidd',description="[optional]it is not supported in swagger ui. copy the url into browser.")
@hidden.route('/download')
class ModelDownload(Resource):
     def get(self):
          if bot==None:
             return {"Error":"Train the Model first and try again."},400
          return send_file(
                "TrainedModel.joblib",
                as_attachment=True,  # Download as an attachment
                 download_name='TrainedModel'# Set the download file name
            )

@model.route("/6.GetDownloadModelLink") 
class linktoModel(Resource):
    def get(self):
        if bot==None:
             return {"Error":"Train the Model first and try again."},400
        if os.path.exists('app/TrainedModel.joblib'):
            download_url = url_for('hidd_model_download', _external=True)
            return {
                 "link":download_url 
            },200
        else:
            abort(404, description=f"File '{'TrainedModel.joblib'}' not found")
#-----------------------cleanUp Apis----------------------------------
space=Namespace("clean-Up APIs",description="clear the space and files")
def clean():
     global data,target,train_data,test_data,dataPrepared,problem,mlModel,bot
     data,target,train_data,test_data,dataPrepared,problem,mlModel,bot=None,None,None,None,None,None,None,None
     if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)  # Delete the file
     if os.path.exists("app/TrainedModel.joblib"):
            os.remove("app/TrainedModel.joblib")
@space.route("/cleanup")
class cleanUp(Resource):
     def get(self):
          if bot==None and not (os.path.exists('app/TrainedModel.joblib'))and not os.path.exists(UPLOAD_FOLDER):
             return {"MESSAGE":"NO DATA TO CLEAN."},200
          clean()
          return{"message":"data cleaned up Successfully."},200









#----------------------------------------updated------------------------------------------------------------





def plot(x,y,name,xl="NA",yl="NA",title="NA",type="bar"):
    if type=='bar':
        plt.figure(figsize=(19, 10))
        plt.bar(x, y)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.title(title)
    else:
         plt.figure(figsize=(19, 19))  # Set the size of the figure
         plt.pie(y, autopct='%d%%', startangle=140)
         plt.legend(x, loc="best")
         plt.title(title)
    plot_path = 'app/static/'+name  # Save the plot in the static folder
    try:
        os.makedirs('static', exist_ok=True)  # Ensure the static folder exists
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        plt.close()
        return f"Error saving plot: {e}"

@hidden.route("/covid")
class covid2(Resource):
     def get(source):
        table_data = [
        {"name": "John Doe", "age": 28, "profession": "Engineer"},
        {"name": "Jane Smith", "age": 34, "profession": "Doctor"},
        {"name": "Emily Davis", "age": 45, "profession": "Lawyer"}
    ]
        response = requests.get('https://api.covidactnow.org/v2/states.json?apiKey=59e19a4762af4c729551915b26418b24')
        data_fin=[]
        for data in response.json():
          dict={}
          dict['state']=data['state']
          dict['population']=data['population']
          dict['last Updated Date']=data['lastUpdatedDate']
          dict['cases till date']=data['actuals']['cases']
          dict['deaths till date']=data['actuals']['deaths']
          dict['Hospital beds capacity']=data['actuals']['hospitalBeds']['capacity']
          dict['current use of Hospbeds']=data['actuals']['hospitalBeds']['currentUsageTotal']
          dict['beds use for covid']=data['actuals']['hospitalBeds']['currentUsageCovid']
          dict['weekly Covid Admissions']=data['actuals']['hospitalBeds']['weeklyCovidAdmissions']
          dict['new Cases']=data['actuals']['newCases']
          dict['new deaths']=data['actuals']['newDeaths']
          dict['vaccines Distributed']=data['actuals']['vaccinesDistributed']
          dict['vaccinations Initiated']=data['actuals']['vaccinationsInitiated']
          dict['vaccinations Completed']=data['actuals']['vaccinationsCompleted']
          data_fin.append(dict)
        df=pd.DataFrame(data=data_fin,columns=data_fin[0].keys())
        for i in range(len(df.columns)):
            if not(df.dtypes[i]=='O'):
              col=df.columns[i]
              mean=df[col].mean()
              df[col].fillna(mean,inplace=True)
        df_html = df.to_html(index=False, classes='table')
        plot(df['state'],df['cases till date'],'cases_state.png',xl='States',yl='Covid cases Till Date',title='Covid Cases In each state',type='bar')
        plot(df['state'],df['deaths till date'],'death_state.png',xl='States',yl='Death cases Till Date',title='Death Cases In each state',type='bar')
        plot(df['state'],df['new deaths'],'new_death_state.png',title='New Death Cases',type='pie')
        resp = make_response(render_template('index.html', df_table=df_html,update=df.iloc[0]['last Updated Date']))
        resp.headers['Content-Type'] = 'text/html'
        return resp
