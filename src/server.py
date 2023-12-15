import os, logging

## Warning supression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

## Warning supression
tf.get_logger().setLevel(logging.ERROR)

from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import random
import numpy as np
import pandas as pd
import pydicom
import keras_ocr
from pydicom.errors import InvalidDicomError
from PIL import Image
import cv2
import json
import os
import shutil
from glob import glob
import datetime
import time
import re
import hashlib


class user_options_class(BaseModel):
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

    if os.path.exists('./session_data/clean/de-identified-files'):
        shutil.rmtree('./session_data/clean/de-identified-files')

    if os.path.isfile('./session_data/user_input.json'):
        os.remove('./session_data/user_input.json')

    if os.path.isfile('./session_data/session.json'):
        os.remove('./session_data/session.json')

    if os.path.isfile('./session_data/requested_action_group_dcm.csv'):
        os.remove('./session_data/requested_action_group_dcm.csv')

    dp, _, fps = list(os.walk('./static/client_data'))[0]
    for fp in fps:
        if fp != '.gitkeep':
            os.remove(dp + '/' + fp)

    dp, _, fps = list(os.walk('./session_data/raw'))[0]
    for fp in fps:
        if fp != '.gitkeep':
            os.remove(dp + '/' + fp)

def DCM2DictMetadata\
(
    ds: pydicom.dataset.Dataset
) \
-> \
pydicom.dataset.Dataset:

    ds_metadata_dict = {}
    for ds_attr in ds:
        ds_tag_idx = re.sub('[(,) ]', '', str(ds_attr.tag))
        if ds_tag_idx == '7fe00010': continue
        if ds_attr.VR != 'SQ':
            value = str(ds_attr.value)
        else:
            value = []
            for inner_ds_idx in range(ds[ds_tag_idx].VM):
                value.append\
                (
                    DCM2DictMetadata\
                    (
                        ds = ds[ds_tag_idx][inner_ds_idx]
                    )
                )

        ds_metadata_dict[ds_tag_idx] = \
        {
            'vr': ds_attr.VR,
            'name': ds_attr.name,
            'value': value,
        }

    return ds_metadata_dict

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

@app.post('/conversion_info')
async def conversion_info\
(
    dicom_pair_fp: List[str] = Body(...)
) \
-> \
dict:

    raw_dcm = pydicom.dcmread(dicom_pair_fp[0])
    cleaned_dcm = pydicom.dcmread(dicom_pair_fp[1])

    raw_img = basic_preprocessing(raw_dcm.pixel_array, downscale = True, toint8 = True, multichannel = True)
    raw_hash = hashlib.sha256(raw_img.tobytes()).hexdigest()
    raw_img_fp = './static/client_data/' + raw_hash + '.png'
    Image.fromarray(raw_img).save(raw_img_fp)

    cleaned_img = basic_preprocessing(cleaned_dcm.pixel_array, downscale = True, toint8 = True, multichannel = True)
    cleaned_hash = hashlib.sha256(cleaned_img.tobytes()).hexdigest()
    cleaned_img_fp = './static/client_data/' + cleaned_hash + '.png'
    Image.fromarray(cleaned_img).save(cleaned_img_fp)

    # with open(file = '../raw_dcm_meta.json', mode = 'w') as file:
    #     json.dump(DCM2DictMetadata(ds = raw_dcm), fp = file)

    # with open(file = '../cleaned_dcm_meta.json', mode = 'w') as file:
    #     json.dump(DCM2DictMetadata(ds = cleaned_dcm), fp = file)

    return \
    {
        'raw_dicom_metadata': DCM2DictMetadata(ds = raw_dcm),
        'raw_dicom_img_fp': raw_img_fp,
        'cleaned_dicom_metadata': DCM2DictMetadata(ds = cleaned_dcm),
        'cleaned_dicom_img_fp': cleaned_img_fp
    }

@app.post('/upload_files/')
async def get_files\
(
    myCheckbox: bool = Form(False),
    files: List[UploadFile] = File(...)
) \
-> \
dict:

    ## Resetting directories
    clean_dirs()

    proper_dicom_paths = []
    total_uploaded_file_bytes = 0
    for file in files:

        ## Serialized file contents
        contents = await file.read()

        fp = './session_data/raw/' + file.filename.split('/')[-1]
        with open(file = fp, mode = 'wb') as f:
            f.write(contents)
        try:
            pydicom.dcmread(fp)
            proper_dicom_paths.append(fp)
        except InvalidDicomError:
            print('W: The following path does not correspond to a DICOM file\n%s'%(fp))
            os.remove(fp)
            print('Irrelevant file deleted')
        total_uploaded_file_bytes += len(contents)
    total_uploaded_file_megabytes = '%.1f'%(total_uploaded_file_bytes / (10**3)**2)

    return {'n_uploaded_files': len(proper_dicom_paths), 'total_size': total_uploaded_file_megabytes}

@app.post('/session')
async def handle_session_button_click\
(
    session_dict: Dict[str, Any]
):
    with open(file = './session_data/session.json', mode = 'w') as file:
        json.dump(session_dict, file)

@app.post('/submit_button')
async def handle_submit_button_click(user_options: user_options_class):

    user_options = dict(user_options)

    dp, _, fps = list(os.walk('./session_data/raw'))[0]
    if set(fps).issubset({'.gitkeep'}):
        return False

    ## ! Update `user_options.json`: Begin

    with open(file = './user_default_options.json', mode = 'r') as file:
        default_options = json.load(file)

    user_options['input_dcm_dp'] = default_options['input_dcm_dp']
    user_options['output_dcm_dp'] = default_options['output_dcm_dp']

    with open(file = './session_data/user_options.json', mode = 'w') as file:
        json.dump(user_options, file)

    ## ! Update `user_options.json`: End

    session, dicom_pair_fps = dicom_deidentifier(SESSION_FP = './session_data/session.json')

    with open(file = './session_data/session.json', mode = 'w') as file:
        json.dump(session, file)

    return dicom_pair_fps

def dicom_deidentifier\
(
    SESSION_FP: None or str = None
) \
-> \
tuple[dict, list[tuple[str]]]:
    '''
        Args:
            SESSION_FP. File path for the session.json file. A session must have the same exact `user_input.json` file independently of interruptions. If `SESSION_FP` is set to None then a new session file will be created at the parent directory.

        Returns:
            session. This dictionary contains the session file's content.
            path_pairs. Contains raw-cleaned DICOM pairs corresponding to the input.
    '''

    ## Initial parameters
    GPU = True ## Set to True if you want to invoke NVIDIA GPU

    if (not GPU):
        tf.config.set_visible_devices([], 'GPU')
        print('[DISABLED] PARALLEL COMPUTATION\n\n---')
    elif len(tf.config.list_physical_devices('GPU')) == 0:
        print('W: No GPU detected, switching to CPU instead')
        print('[DISABLED] PARALLEL COMPUTATION\n\n---')
    elif tf.config.list_physical_devices('GPU')[0][1] == 'GPU':
        print('[ENABLED] PARALLEL COMPUTATION\n\n---')

    ## Parse all possible DICOM metadata configurations
    action_groups_df = pd.read_csv(filepath_or_buffer = './action_groups/action_groups_dcm.csv', index_col = 0)

    if SESSION_FP == None or not os.path.isfile(SESSION_FP):
        print('Creating a new session')
        session = dict()
    else:
        with open(file = './session_data/session.json', mode = 'r') as file:
            print('Parsing already generated session')
            session = json.load(file)

    if os.path.isfile('./session_data/user_options.json'):
        with open(file = './session_data/user_options.json', mode = 'r') as file:
            user_input = json.load(file)
    else:
        exit('E: No client de-identification configuration was provided')

    pseudo_patient_ids = []
    for patient_deidentification_properties in session.values():
        pseudo_patient_ids.append(int(patient_deidentification_properties['patientPseudoId']))

    if pseudo_patient_ids == []:
        max_pseudo_patient_id = -1
    else:
        max_pseudo_patient_id = max(pseudo_patient_ids)

    ## Get one input DICOM
    rw_obj = rwdcm(in_dp = user_input['input_dcm_dp'], out_dp = user_input['output_dcm_dp'])

    while next(rw_obj):

        dcm = rw_obj.parse_file()
        if dcm == False:
            print('File already converted\nSkipping\n---\n')
            continue

        print('Processing DICOM file')

        ## User input parameter validity check
        date_processing_choices = {'keep', 'offset', 'remove'}
        assert user_input['date_processing'] in date_processing_choices, 'E: Invalid date processing input'

        real_patient_id = dcm[0x0010, 0x0020].value
        patient_deidentification_properties = session.get(real_patient_id, False)
        if not patient_deidentification_properties:
            max_pseudo_patient_id += 1
            session[real_patient_id] = {'patientPseudoId': '%.6d'%max_pseudo_patient_id}
            days_total_offset = round(random.uniform(10 * 365, (2 * 10) * 365))
            seconds_total_offset = round(random.uniform(0, 24 * 60 * 60))
        else:
            days_total_offset = session[real_patient_id]['daysOffset']
            seconds_total_offset = session[real_patient_id]['secondsOffset']

        ## Define metadata action group based on user input
        requested_action_group_df = get_action_group(user_input = user_input, action_groups_df = action_groups_df)

        requested_action_group_df.to_csv('./session_data/requested_action_group_dcm.csv')

        ## Adjusts DICOM metadata based on user parameterization
        dcm, tag_value_replacements = adjust_dicom_metadata\
        (
            user_input = user_input,
            dcm = dcm,
            action_group_fp = './session_data/requested_action_group_dcm.csv',
            patient_pseudo_id = session[real_patient_id]['patientPseudoId'],
            days_total_offset = days_total_offset,
            seconds_total_offset = seconds_total_offset
        )

        ## Session update
        session[real_patient_id]['daysOffset'] = tag_value_replacements['days_total_offset']
        session[real_patient_id]['secondsOffset'] = tag_value_replacements['seconds_total_offset']

        ## Adds metadata tag with field that specifies the anonymization process
        dcm = deidentification_attributes(user_input = user_input, dcm = dcm)

        if user_input['clean_image']:
            ## Cleans burned in text in pixel data
            dcm, _, _ = keras_ocr_dicom_image_text_remover(dcm = dcm)

        print('DICOM Processing Completed')

        ## Store DICOM file and create output directories
        rw_obj.export_processed_file(dcm = dcm)

        ## Overwrite session file to save progress
        rw_obj.export_session(session = session)

    print('Operation completed')

    return session, rw_obj.dicom_pair_fps

def deidentification_attributes(user_input: dict, dcm: pydicom.dataset.FileDataset) -> pydicom.dataset.FileDataset:
    '''
        Description:
            Appends additional DICOM attributes that specify the anonymization process as per Table CID 7050 from https://dicom.nema.org/medical/dicom/2019a/output/chtml/part16/sect_CID_7050.html. Additional relevant information can be found at https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_E.3.html.

        Args:
            user_input.
            dcm.

        Returns
            dcm_. Contains more tags that specifies the anonymization procedure.
    '''

    user_input_lookup_table = \
    {
        'clean_image': '113101',
        'retain_safe_private': '113111',
        'retain_uids': '113110',
        'retain_device_identity': '113109',
        'retain_patient_characteristics': '113108',
        'date_processing':
        {
            'offset': '113107',
            'keep': '113106',
        },
        'retain_descriptors': '113105',
    }

    DeIdentificationCodeSequence = 'DCM:11310'

    assert set(user_input_lookup_table.keys()).issubset(set(user_input.keys())), 'E: Inconsistency with user input keys with lookup de-identification table keys'
    for OptionName, DeIdentificationCodes in user_input_lookup_table.items():
        choice = user_input[OptionName]
        if OptionName == 'date_processing':
            if choice in user_input_lookup_table['date_processing']:
                DeIdentificationCodeSequence += '/' + user_input_lookup_table['date_processing'][choice]
        else:
            if choice:
                DeIdentificationCodeSequence += '/' + user_input_lookup_table[OptionName]

    ## (0x0012, 0x0062) -> Patient Identity Removed
    dcm.add_new\
    (
        tag = (0x0012, 0x0062),
        VR = 'LO',
        value = 'YES'
    )

    ## (0x0012, 0x0063) -> De-identification Method
    dcm.add_new\
    (
        tag = (0x0012, 0x0063),
        VR = 'LO',
        value = DeIdentificationCodeSequence
    )

    if user_input['clean_image']:
        ## (0x0028, 0x0301) -> Burned In Annotation
        dcm.add_new\
        (
            tag = (0x0028, 0x0301),
            VR = 'LO',
            value = 'NO'
        )

    return dcm

def ndarray_size(arr: np.ndarray) -> int:
    return arr.itemsize*arr.size

def basic_preprocessing(img, downscale, toint8 = True, multichannel = True) -> np.ndarray:
    '''
        Description:
            Main preprocessing. It is imperative that the image is converted to (1) uint8 and in (2) RGB in order for keras_ocr's detector to properly function.

        Args:
            downscale. Bool.

        Returns:
            out_image. Its shape is (H, W) if `multichannel` is set to `False`, otherwise its shape is (H, W, 3).
    '''

    if downscale:
        ## Downscale
        downscale_dimensionality = 1024
        new_shape = (min([downscale_dimensionality, img.shape[0]]), min([downscale_dimensionality, img.shape[1]]))
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
        # print('Image downscaled to (%d, %d)'%(new_shape[0], new_shape[1]))

    if toint8:
        img = (255.0 * (img / np.max(img))).astype(np.uint8)

    if (multichannel) and (len(img.shape) == 2):
        img = np.stack(3*[img], axis = -1)

    return img

def text_remover(img, bboxes: np.ndarray, initial_array_shape, downscaled_array_shape):
    '''
        Args:
            bboxes. Shape (n_bboxes, 4, 2), where 4 is the number of vertices for each box and 2 are the plane coordinates. The vertices inside the bboxes array should be sorted in a way that corresponds to a geometrically counter-clockwise order. For example given a non-rotated (0 degree) bounding box with index 0, the following rule applies
                bboxes[0, 0, :] -> upper left vertex
                bboxes[0, 1, :] -> lower left vertex
                bboxes[0, 2, :] -> lower right vertex
                bboxes[0, 3, :] -> upper right vertex
    '''

    reducted_region_color = np.mean(img).astype(np.uint16)

    multiplicative_mask = np.ones(downscaled_array_shape, dtype = np.uint8)
    additive_mask = np.zeros(initial_array_shape, dtype = np.uint8)

    ## Concecutive embeddings of bounding boxes
    for bbox in bboxes:

        x0, y0 = bbox[0, 0:(1+1)]
        x1, y1 = bbox[1, 0:(1+1)]
        x2, y2 = bbox[2, 0:(1+1)]
        x3, y3 = bbox[3, 0:(1+1)]

        rectangle = np.array\
        (
            [
                [
                    [x0, y0],
                    [x1, y1],
                    [x2, y2],
                    [x3, y3]
                ]
            ],
            dtype = np.int32 ## Must remain this way. Otherwise, cv2.fillPoly will throw an error.
        )

        ## Filled rectangle
        cv2.fillPoly(multiplicative_mask, rectangle, 0)

    ## When multiplied with image, bounding box pixels will be replaced with 0
    multiplicative_mask = cv2.resize(multiplicative_mask, (initial_array_shape[1], initial_array_shape[0]), interpolation = cv2.INTER_NEAREST)

    ## When added after multiplication, bounding box pixels will be replaced with 255
    additive_mask = reducted_region_color * (multiplicative_mask == 0)

    img_ = img.copy()
    img_ = (img_ * multiplicative_mask + additive_mask)

    return img_

def keras_ocr_dicom_image_text_remover(dcm):

    def prep_det_keras_ocr(img):

        img_prep = basic_preprocessing(img = img, downscale = True)
        bboxes = det_keras_ocr(img_prep)

        return img_prep, bboxes

    def det_keras_ocr(img):

        pipeline = keras_ocr.detection.Detector()

        ## Returns a ndarray with shape (n_bboxes, 4, 2) where 4 is the number of points for each box, 2 are the plane coordinates.
        bboxes = pipeline.detect([img])[0]

        return bboxes

    t0 = time.time()

    ## Extract image data from dicom files
    ## Scalar data type -> uint16
    dcm.decompress()
    raw_img_uint16_grayscale = dcm.pixel_array

    ## Secondary information about the DICOM file
    print('Input DICOM file information')
    print('Input image shape: ', raw_img_uint16_grayscale.shape)

    t1 = time.time()

    raw_img_uint8_grayscale, bboxes = prep_det_keras_ocr(img = raw_img_uint16_grayscale)

    removal_period = time.time() - t1

    initial_array_shape = raw_img_uint16_grayscale.shape
    downscaled_array_shape = raw_img_uint8_grayscale.shape[:-1]

    if np.size(bboxes) != 0:

        cleaned_img = text_remover\
        (
            img = raw_img_uint16_grayscale,
            bboxes = bboxes,
            initial_array_shape = initial_array_shape,
            downscaled_array_shape = downscaled_array_shape
        )

        ## Update the DICOM image data with the modified image
        dcm.PixelData = cleaned_img.tobytes()

    else:

        print('Image state: No text detected')

    total_period = time.time() - t0

    return dcm, removal_period, total_period

def get_action_group(user_input: dict, action_groups_df: pd.DataFrame) -> pd.DataFrame:
    '''
        Description:
            Depending on the user's choice, and an action group lookup table, based on Nema's action principles as per tables Table E.1-1a and Table E.1-1 of https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_e.html, an action group is generated. That action group is in the form of a column from Table E.1-1 acting as a configuration basis for the anonymization of a DICOM file.

        Args:
            user_input. The user's anonymization process. Sufficient property:
            - Contains all of the following items
                'clean_image': Value type bool
                'retain_safe_private': Value type bool
                'retain_uids': Value type bool
                'retain_device_identity': Value type bool
                'retain_patient_characteristics': Value type bool
                'date_processing': Value type str
                'retain_descriptors': Value type bool
            action_groups_df. Lookup table for action groups. A modified version of Table E.1-1.

        Returns:
            requested_action_group_df. Requested action group, defined by the user's choice. Each row corresponds to an attribute. The index column contains tag codes. Column 0 contains the tag names. Column 2 contains the action code.
    '''

    def merge_action(primary_df, Action2BeMerged_df):

        return primary_df.where\
        (
            cond = Action2BeMerged_df.isna(),
            other = Action2BeMerged_df,
            axis = 0,
            inplace = False,
        )

    requested_action_group_df = pd.DataFrame\
    (
        data = action_groups_df['Default'].to_list(),
        columns = ['Requested Action Group'],
        index = action_groups_df.index
    )

    requested_action_group_df.insert\
    (
        loc = 0,
        column = 'Name',
        value = action_groups_df['Name'].to_list(),
    )

    if user_input['retain_safe_private']:

        requested_action_group_df['Requested Action Group'] = merge_action(primary_df = requested_action_group_df['Requested Action Group'], Action2BeMerged_df = action_groups_df['Rtn. Safe Priv. Opt.'])

    if user_input['retain_uids']:

        requested_action_group_df['Requested Action Group'] = merge_action(primary_df = requested_action_group_df['Requested Action Group'], Action2BeMerged_df = action_groups_df['Rtn. UIDs Opt.'])

    if user_input['retain_device_identity']:

        requested_action_group_df['Requested Action Group'] = merge_action(primary_df = requested_action_group_df['Requested Action Group'], Action2BeMerged_df = action_groups_df['Rtn. Dev. Id. Opt.'])

    if user_input['retain_patient_characteristics']:

        requested_action_group_df['Requested Action Group'] = merge_action(primary_df = requested_action_group_df['Requested Action Group'], Action2BeMerged_df = action_groups_df['Rtn. Pat. Chars. Opt.'])

    if user_input['date_processing'] == 'keep':

        requested_action_group_df['Requested Action Group'] = merge_action(primary_df = requested_action_group_df['Requested Action Group'], Action2BeMerged_df = action_groups_df['Rtn. Long. Modif. Dates Opt.'])

    elif user_input['date_processing'] == 'offset':

        requested_action_group_df['Requested Action Group'] = merge_action(primary_df = requested_action_group_df['Requested Action Group'], Action2BeMerged_df = action_groups_df['Offset Long. Modif. Dates Opt.'])

    elif user_input['date_processing'] == 'remove':

        requested_action_group_df['Requested Action Group'] = merge_action(primary_df = requested_action_group_df['Requested Action Group'], Action2BeMerged_df = action_groups_df['Remove Long. Modif. Dates Opt.'])

    if user_input['retain_descriptors']:

        requested_action_group_df['Requested Action Group'] = merge_action(primary_df = requested_action_group_df['Requested Action Group'], Action2BeMerged_df = action_groups_df['Rtn. Desc. Opt.'])

    return requested_action_group_df

def adjust_dicom_metadata(user_input: dict, dcm: pydicom.dataset.FileDataset, action_group_fp: str, patient_pseudo_id: str, days_total_offset, seconds_total_offset) -> (pydicom.dataset.FileDataset, dict):
    '''
        Description:
            Applies an action group on a DICOM file's metadata.

        Args:
            user_input. The user's anonymization process.
            dcm. DICOM object.
            action_group_fp. Path for .csv file that contains the an action group. The file's content is saved in action_group_df.
                action_group_df. Specifies how exactly the DICOM's metadata will be modified according to its attribute actions.
            patient_pseudo_id.

        Returns:
            updated_dcm. DICOM object adjusted according to configuration.
            tag_value_replacements. Contains things like patient pseudo ID and date offset.
    '''

    def get_pseudo_date(input_date_str, days_total_offset):

        input_date = datetime.datetime.strptime(input_date_str, '%Y%m%d')
        output_date = input_date + datetime.timedelta(days = days_total_offset)
        output_date_str = output_date.strftime('%Y%m%d')

        return output_date_str

    def get_pseudo_day_time(seconds_total_offset):

        output_hours = seconds_total_offset // 3600
        output_minutes = (seconds_total_offset % 3600) // 60
        output_seconds = (seconds_total_offset % 3600) % 60

        output_time_str = '%.2d%.2d%.2d'%(output_hours, output_minutes, output_seconds)

        return output_time_str

    def clean_one_attribute_on_one_dataset_level(ds: pydicom.dataset.Dataset, action: str, action_attr_tag_idx: str) -> pydicom.dataset.Dataset:

        for ds_attr in ds:
            ds_tag_idx = re.sub('[(,) ]', '', str(ds_attr.tag))
            if ds[ds_tag_idx].VR == 'SQ':
                for inner_ds_idx in range(ds[ds_tag_idx].VM):
                    ds[ds_tag_idx].value[inner_ds_idx] = clean_one_attribute_on_one_dataset_level\
                    (
                        ds = ds[ds_tag_idx][inner_ds_idx],
                        action = action,
                        action_attr_tag_idx = action_attr_tag_idx
                    )

            ## Leaf node
            elif action_attr_tag_idx == ds_tag_idx:

                if action == 'Z':

                    assert ds_tag_idx in ['00100010', '00100020'], 'E: Cannot apply action code `Z` in any other attribute besides Patient ID and Patient Name; the issue is likely on the action group config object'

                    ds[ds_tag_idx].value = patient_pseudo_id

                elif action == 'X':

                    ds[ds_tag_idx].value = ''

                elif action == 'C':

                    if ds[ds_tag_idx].VR == 'DA':

                        tag_value_replacements['days_total_offset'] = days_total_offset

                        ds[ds_tag_idx].value = get_pseudo_date(input_date_str = ds[ds_tag_idx].value, days_total_offset = days_total_offset)

                    ## W: Significant temporal information is wasted as it is not implemented using an offset that is added to the corresponding timestamp value; it instead substitutes a common random value; this was implemented like this to avoid potential 24-hour overflows, which would potentially lead to an increase of some datetime day by 1, but from which DA tag?
                    elif ds[ds_tag_idx].VR == 'TM':

                        tag_value_replacements['seconds_total_offset'] = seconds_total_offset

                        ds[ds_tag_idx].value = get_pseudo_day_time(seconds_total_offset = tag_value_replacements['seconds_total_offset'])

        return ds

    action_group_df = pd.read_csv(filepath_or_buffer = action_group_fp, index_col = 0)

    tag_value_replacements = dict()

    ## Will be replaced only if date and time offsets are applied to at least one tag
    tag_value_replacements['days_total_offset'] = 0
    tag_value_replacements['seconds_total_offset'] = 0

    days_total_offset_was_applied_at_least_once = False
    seconds_total_offset_was_applied_at_least_once = False

    # print('Debugging section: 0xad4a9cb412. Remove some lines directly below.')
    # print('Creating a dummy sequence with multiple recursive sequences of multiple datasets each, containing values that should be removed.')
    # for dcm_attr in dcm:
    #     dcm_tag_idx = re.sub('[(,) ]', '', str(dcm_attr.tag))
    #     if dcm[dcm_tag_idx].VR == 'SQ':
    #         # dcm[dcm_tag_idx].value.append(dcm[dcm_tag_idx][0])
    #         ds = pydicom.dataset.Dataset()
    #         ds.PatientName = "CITIZEN^Joan"
    #         ds.add_new(0x00100020, 'LO', '12345')
    #         ds[0x0008, 0x0022] = pydicom.DataElement(0x00080022, 'DA', '20010101')
    #         ds.add_new((0x0008, 0x1032), 'SQ', pydicom.sequence.Sequence())
    #         ds[0x0008, 0x1032].value = [pydicom.dataset.Dataset()]
    #         ds[0x0008, 0x1032][0].add_new(0x00100010, 'LO', 'CITIZEN^Joan')
    #         ds[0x0008, 0x1032][0].add_new(0x00100020, 'LO', 'WTF')
    #         dcm[dcm_tag_idx].value = [ds, ds]

    ## Cleaning primary DICOM
    ## Scans all DICOM tags until it intercepts `action_attr_tag_idx`. Then it updates that exact tag index per dataset. You can view this recursion statically, as an arbitrary directed graph.
    ## Each leaf node counts as one attribute update. Collectively this can alter multiple tags of the DICOM object.
    for action_attr_tag_idx in action_group_df.index:
        action = action_group_df.loc[action_attr_tag_idx].iloc[1]
        dcm = clean_one_attribute_on_one_dataset_level\
        (
            ds = dcm,
            action = action,
            action_attr_tag_idx = action_attr_tag_idx
        )

    return dcm, tag_value_replacements

class rwdcm:
    '''
        Description:
            Can read and write multiple files on a directory.
    '''

    def __init__(self, in_dp: str, out_dp: str):

        self.SAFETY_SWITCH = True
        if not self.SAFETY_SWITCH:
            print('W: Safety switch is off. Output directory can now be deleted.')

        if in_dp[-1] != '/': in_dp = in_dp + '/'
        self.raw_data_dp = in_dp
        self.raw_dicom_paths = sorted(self.get_dicom_paths(data_dp = self.raw_data_dp))
        self.dicom_pair_fps = []
        self.clean_data_dp = out_dp + '/' + 'de-identified-files/'

        already_cleaned_dicom_paths = self.get_dicom_paths(data_dp = self.clean_data_dp)
        self.hashes_of_already_converted_files = [already_cleaned_dicom_path.split('/')[-1].split('.')[0] for already_cleaned_dicom_path in already_cleaned_dicom_paths]

        self.n_dicom_files = len(self.raw_dicom_paths)

        print('\nTotal number of DICOM files existing inside the input directory:\n%d'%(self.n_dicom_files))
        print('---', end = 2 * '\n')

        self.DICOM_IDX = -1

    def __next__(self):

        self.DICOM_IDX += 1
        if self.DICOM_IDX <= self.n_dicom_files - 1:
            self.raw_dicom_path = self.raw_dicom_paths[self.DICOM_IDX]
            print('---\n')
            print('DICOM List Index:', self.DICOM_IDX)
            return True
        else:
            return False

    def get_dicom_paths(self, data_dp):

        dicom_paths = \
        (
            glob\
            (
                pathname = data_dp + '*',
                recursive = True
            )
        )

        proper_dicom_paths = []
        for dicom_path in dicom_paths:
            try:
                pydicom.dcmread(dicom_path)
                proper_dicom_paths.append(dicom_path)
            except InvalidDicomError:
                continue

        return proper_dicom_paths

    def parse_file(self):

        self.input_dicom_hash = hashlib.sha256(self.raw_dicom_path.encode('UTF-8')).hexdigest()

        if self.input_dicom_hash in self.hashes_of_already_converted_files:
            return False
        else:
            dcm = pydicom.dcmread(self.raw_dicom_path)
            print('Parsed\n%s'%(self.raw_dicom_path))
            return dcm

    def export_processed_file(self, dcm):

        self.clean_dicom_dp = self.clean_data_dp + str(dcm[0x0010, 0x0020].value) + '/' + str(dcm[0x0008, 0x0060].value) + '/' + str(dcm[0x0020, 0x0011].value)
        if not os.path.exists(self.clean_dicom_dp):
            os.makedirs(self.clean_dicom_dp)

        clean_dicom_fp = self.clean_dicom_dp + '/' + self.input_dicom_hash + '.dcm'

        print('Exporting file at\n%s'%(clean_dicom_fp))

        dcm.save_as(clean_dicom_fp)

        self.dicom_pair_fps.append((self.raw_dicom_path, clean_dicom_fp))

    def export_session(self, session):

        with open(self.clean_data_dp + '/session.json', 'w') as file:
            json.dump(session, file)