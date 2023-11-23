import pandas as pd
import numpy as np
import pydicom
import re
import datetime
import time


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

    action_group_df = pd.read_csv(filepath_or_buffer = action_group_fp, index_col = 0)

    tag_value_replacements = dict()

    ## [ToDo] Make this work for sessions - Pseudo time and pseudo date. In case we need to add offsets to temporal values. Note that the resulting application of these random values is conducted in a way that it preserves temporal relationships (i.e. longitudinal temporal information).


    ## Will be replaced only if date and time offsets are applied to at least one tag
    tag_value_replacements['days_total_offset'] = 0
    tag_value_replacements['seconds_total_offset'] = 0

    days_total_offset_was_applied_at_least_once = False
    seconds_total_offset_was_applied_at_least_once = False

    for action_attr_tag_idx in action_group_df.index:
        action = action_group_df.loc[action_attr_tag_idx].iloc[1]
        for dcm_attr in dcm:
            dcm_tag_idx = re.sub('[(,) ]', '', str(dcm_attr.tag))
            if action_attr_tag_idx == dcm_tag_idx:

                if dcm[dcm_tag_idx].VR == 'SQ': continue

                if action == 'Z':

                    assert dcm_tag_idx in ['00100010', '00100020'], 'E: Cannot apply action code `Z` in any other attribute besides Patient ID and Patient Name'

                    dcm[dcm_tag_idx].value = patient_pseudo_id

                elif action == 'X':

                    dcm[dcm_tag_idx].value = ''

                elif action == 'C':

                    if dcm[dcm_tag_idx].VR == 'DA':

                        dcm[dcm_tag_idx].value = get_pseudo_date(dcm[dcm_tag_idx].value, days_total_offset = days_total_offset)

                        tag_value_replacements['days_total_offset'] = days_total_offset

                    elif dcm[dcm_tag_idx].VR == 'TM':

                        dcm[dcm_tag_idx].value = get_pseudo_day_time(seconds_total_offset = tag_value_replacements['seconds_total_offset'])

                        tag_value_replacements['seconds_total_offset'] = seconds_total_offset

    return dcm, tag_value_replacements