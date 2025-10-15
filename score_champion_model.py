# script to score whichever model is the champion based on model version set


# for jobs with a conda environment - generalml includes ads and sklearn
# for byoc jobs, we can simply add these as dependencies

# load dependencies
import ads
import tempfile
from ads.model import SklearnModel
from ads.model import ModelVersionSet
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from ads.common.model_metadata import UseCaseType
import json
import joblib
from ads.common.auth import default_signer
import logging


logging.info('starting job')

# set ads auth to resource principal
ads.set_auth('resource_principal')

# set ocids
project_ocid = 'ocid1.datascienceproject...'
compartment_ocid = 'ocid1.compartment.oc1....'

# sync with model set
logging.info('fetching model version set metadata')
mvs = ModelVersionSet.from_name(name='demo-model-version-set',compartment_id=compartment_ocid)

# NOTE: if we had more than one model version set in this compartment we would need to filter it further

# find model id for model with label==champion

logging.info('finding champion model')
models_list = mvs.models()

def find_champion(models_list):
    for i in range(len(models_list)):
        model = models_list[i]
        if model.version_label == 'Champion Model':
            msg = str('model '+str(i)+' is the champion')
            logging.info(msg)
            champion_id = model.id
    return champion_id

champion = find_champion(models_list)

# load champion model

logging.info('downloading champion model artefact')
temp_dir = tempfile.mkdtemp()
downloaded_model = SklearnModel.from_model_catalog(champion,artifact_dir=temp_dir,ignore_conda_error=True)    

model = joblib.load(temp_dir+'/model.joblib')

# load dataset from object storage
logging.info('fetching hold-out data from object storage')
bucket_name = 'model-version-sets'
file_name = 'data/x_test.csv'
namespace = '<your-namespace>'
df = pd.read_csv(f"oci://{bucket_name}@{namespace}/{file_name}", storage_options=default_signer())

# create predictions
logging.info('making predictions from champion model')
preds = model.predict(df)

# write predictions back to object storage
logging.info('writing results to object storage')
file_name = 'outputs/job_preds.csv'
pd.DataFrame(preds).to_csv(f"oci://{bucket_name}@{namespace}/{file_name}", storage_options=default_signer(),index=False)

logging.info('job complete')