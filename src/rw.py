'''
    Description:
        Read/write related functionality.
'''


import pydicom
from pydicom.errors import InvalidDicomError
from PIL import Image
import numpy as np
import shutil
import os
import json
from glob import glob
import hashlib


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
            print('List index:', self.DICOM_IDX)
            print('---\n')
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

    def export_session(self, session):

        with open(self.clean_data_dp + '/session.json', 'w') as file:
            json.dump(session, file)