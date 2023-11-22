import pandas as pd
import rw
import pydicom
import json
import action_tools

def encoded_anonymization_attributes(user_input: dict, dcm: pydicom.dataset.FileDataset) -> pydicom.dataset.FileDataset:
    '''
        Description:
            Adds additional DICOM attributes that specify the anonymization process as per Table CID 7050 from https://dicom.nema.org/medical/dicom/2019a/output/chtml/part16/sect_CID_7050.html. Additional relevant information can be found at https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_E.3.html.

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


def main(session = None):

    ## ! Initial anonymization parameters: Begin

    ## Parse internal configs
    action_groups_df = pd.read_csv(filepath_or_buffer = '../action_groups_dcm.csv', index_col = 0)

    with open(file = '../user_default_input.json', mode = 'r') as file:
        user_input = json.load(file)

    ## Replace with POST request taken from form's inputs
    with open(file = '../user_input.json', mode = 'r') as file:
        user_input = json.load(file)

    ## Get one input DICOM
    dcm = pydicom.dcmread(fp = user_input['input_dcm_fp'])

    ## ! Initial anonymization parameters: End

    ## User input parameter validity check
    date_processing_choices = {'keep', 'offset', 'remove'}
    assert user_input['date_processing'] in date_processing_choices, 'E: Invalid date processing input'

    ## Define metadata action group based on user input
    requested_action_group_df = action_tools.get_action_group(user_input = user_input, action_groups_df = action_groups_df)

    ## [ToDo] Create session file IF there is no other depending on another user input. The user will be asked right when the session file will be searched by the executable
    requested_action_group_df.to_csv('../requested_action_group_dcm.csv')

    ## Adjusts DICOM metadata based on user parameterization
    dcm, tag_value_replacements = action_tools.adjust_dicom_metadata\
    (
        user_input = user_input,
        dcm = dcm,
        action_group_fp = '../requested_action_group_dcm.csv'
    )

    ## Adds metadata tag with field that specifies the anonymization process
    dcm = encoded_anonymization_attributes(user_input = user_input, dcm = dcm)

    ## Create proper session file
    ## Add date if date setting is true
    ## Add time if time settings is true

    ## Create a hash value and export object in file
    # dcm.save_as(user_input['output_dcm_fp'], write_like_original = False)
    pydicom.filewriter.dcmwrite(filename = user_input['output_dcm_fp'], dataset = dcm, write_like_original = False)


if __name__ == '__main__':

    main()