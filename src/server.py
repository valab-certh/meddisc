from typing import Annotated

from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import pydicom
import json
import dcm_deidentifier
import os
import shutil


class user_input_class(BaseModel):
    ## These have to match exactly with javascript's "dictionary" keys, both the keys and the data types
    clean_image: bool
    retain_safe_private: bool
    retain_uids: bool
    retain_device_identity: bool
    retain_device_identity: bool
    retain_patient_characteristics: bool
    date_processing: str
    retain_descriptors: bool
    patient_pseudo_id_prefix: str

class session_patient_instance_class(BaseModel):
    patientPseudoId: str
    daysOffset: float
    secondsOffset: int

class session_class(BaseModel):
    patients: dict[str, session_patient_instance_class]


def clean_dirs():

    if os.path.exists('../session_data/clean/de-identified-files'):
        shutil.rmtree('../session_data/clean/de-identified-files')

    # if os.path.isfile('../session_data/clean/de-identified-files.zip'):
    #     os.remove('../session_data/clean/de-identified-files.zip')

    if os.path.isfile('../session_data/user_input.json'):
        os.remove('../session_data/user_input.json')

    if os.path.isfile('../session_data/session.json'):
        os.remove('../session_data/session.json')

    dp, _, fps = list(os.walk('../session_data/raw'))[0]
    for fp in fps:
        if fp != '.gitkeep':
            os.remove(dp + '/' + fp)

app = FastAPI()
app.mount\
(
    path = '/static',
    app = StaticFiles(directory='static'),
    name = 'sttc'
)

@app.get('/')
async def get_root():
    return FileResponse('./static/index.html')

@app.post('/upload_files/')
async def get_files\
(
    myCheckbox: bool = Form(False),
    files: List[UploadFile] = File(...)
):

    ## Resetting directories
    clean_dirs()
    if os.path.isfile('../session_data/clean/de-identified-files.zip'):
        os.remove('../session_data/clean/de-identified-files.zip')

    total_uploaded_file_bytes = 0
    files_content = []
    for file in files:
        ## Serialized file contents
        contents = await file.read()
        files_content.append(contents)
        with open(file = '../session_data/raw/' + file.filename.split('/')[-1], mode = 'wb') as f:
            f.write(contents)
        total_uploaded_file_bytes += len(contents)
    total_uploaded_file_megabytes = '%.1f'%(total_uploaded_file_bytes / (10**3)**2)

    return {"n_uploaded_files": len(files), "total_size": total_uploaded_file_megabytes}

@app.post('/session')
async def handle_session_button_click(session_dict: Dict[str, Any]):
    with open(file = '../session_data/session.json', mode = 'w') as file:
        json.dump(session_dict, file)

@app.post('/submit_button_clicked')
async def handle_submit_button_click(user_input_dict: user_input_class):

    requested_parameters = dict(user_input_dict)

    ## ! Update `user_input.json`: Begin

    with open(file = '../user_default_input.json', mode = 'r') as file:
        user_input = json.load(file)

    for requested_parameter_key, requested_parameter_value in requested_parameters.items():
        user_input[requested_parameter_key] = requested_parameter_value

    with open(file = '../session_data/user_input.json', mode = 'w') as file:
        json.dump(user_input, file)

    ## ! Update `user_input.json`: End

    session = dcm_deidentifier.dicom_deidentifier(SESSION_FP = '../session_data/session.json')

    with open(file = '../session_data/session.json', mode = 'w') as file:
        json.dump(session, file)

    shutil.make_archive(base_name = user_input['output_dcm_dp'] + '/de-identified-files', format = 'zip', root_dir = user_input['output_dcm_dp'] + '/de-identified-files')

    clean_dirs()

    return FileResponse(path = user_input['output_dcm_dp'] + '/de-identified-files.zip', media_type = 'application/octet-stream')