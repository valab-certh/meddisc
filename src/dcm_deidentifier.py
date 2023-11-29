import os, logging

## Warning supression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

## Warning supression
tf.get_logger().setLevel(logging.ERROR)

import pandas as pd
import rw
import pydicom
import json
import random

import action_tools
import dcm_pixel_data_cleaner


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

def dicom_deidentifier(SESSION_FP: None or str = None):
    '''
        Args:
            SESSION_FP. File path for the session.json file. A session must have the same exact `user_input.json` file independently of interruptions. If None then a new session file will be created at the parent directory.
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

    if os.path.isfile('./session_data/user_input.json'):
        with open(file = './session_data/user_input.json', mode = 'r') as file:
            user_input = json.load(file)
    else:
        print('W: No client de-identification configuration was provided; overriding default de-identification settings')
        with open(file = './user_default_input.json', mode = 'r') as file:
            user_input = json.load(file)

    pseudo_patient_ids = []
    for patient_deidentification_properties in session.values():
        pseudo_patient_ids.append(int(patient_deidentification_properties['patientPseudoId']))

    if pseudo_patient_ids == []:
        max_pseudo_patient_id = -1
    else:
        max_pseudo_patient_id = max(pseudo_patient_ids)

    ## Get one input DICOM
    rw_obj = rw.rwdcm(in_dp = user_input['input_dcm_dp'], out_dp = user_input['output_dcm_dp'])

    while next(rw_obj):

        dcm = rw_obj.parse_file()
        if dcm == False:
            print('File already converted\nSkipping\n---\n')
            continue

        print('Starting to process the raw DICOM\'s object')

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
        requested_action_group_df = action_tools.get_action_group(user_input = user_input, action_groups_df = action_groups_df)

        requested_action_group_df.to_csv('./session_data/requested_action_group_dcm.csv')

        ## Adjusts DICOM metadata based on user parameterization
        dcm, tag_value_replacements = action_tools.adjust_dicom_metadata\
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
            dcm, _, _ = dcm_pixel_data_cleaner.keras_ocr_dicom_image_text_remover(dcm = dcm)

        ## Store DICOM file and create output directories
        rw_obj.export_processed_file(dcm = dcm)

        ## Overwrite session file to save progress
        rw_obj.export_session(session = session)

    print('Operation completed')

    return session


if __name__ == '__main__':

    dicom_deidentifier()