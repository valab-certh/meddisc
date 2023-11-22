'''
    Description:
        Read/write related functionality.
'''


import pydicom as dicom
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import shutil
import os
from glob import glob


class rw_2_dcm:
    '''
        Description:
            Can read and write multiple files on a directory. Given a directory path (e.g. '../dataset/raw/direc'), it
            1. Copies the directory structure along with all non-DICOM files inside '../dataset/clean'.
            2. Recursively searches all DICOM files inside the input directory.
            3. Parses all found DICOM files from inside the input directory, and pastes them in the respective paths of the output directory after a potential modification. If the output DICOM path already contains a DICOM file, then it skips it.
    '''

    def __init__(self, dp: str):
        '''
            Args:
                dp: Directory path of the input directory.
        '''

        self.modifiable_file_extension_names = \
        [
            'dcm',
            'dcM',
            'dCm',
            'dCM',
            'Dcm',
            'DcM',
            'DCm',
            'DCM'
        ]

        self.SAFETY_SWITCH = True
        if not self.SAFETY_SWITCH:
            print('W: Safety switch is off. Output directory can now be deleted.')

        if dp[-1] != '/': dp = dp + '/'
        self.raw_data_dp = dp
        self.clean_data_dp_ = '../dataset/clean/'
        self.clean_data_dp_last_dir_idx = len(self.raw_data_dp.split('/')) - 2
        self.clean_data_dp = self.clean_data_dp_ + self.raw_data_dp.split('/')[self.clean_data_dp_last_dir_idx] + '/'
        self.raw_dicom_paths = sorted(self.generate_dicom_paths(data_dp = self.raw_data_dp))

        self.n_dicom_files = len(self.raw_dicom_paths)

        self.copy_dir_structure()

        print('Total number of DICOM files existing inside the input directory:\n%d'%(self.n_dicom_files))
        print('---', end = 2 * '\n')

        overwrite_choices = {'O': 'Overwrite', 'I': 'Ignore'}
        inp = input('About the DICOM files that already exist inside the output directory. If they have the same name as the corresponding input. How should they be handled?\n[I] Ignore DICOM files\n[O] Overwrite DICOM files with newly processed ones\n> ')
        while inp not in overwrite_choices.keys():
            inp = input('Select a valid input:\n> ')
        print('Proceeding with:', overwrite_choices[inp])
        self.overwrite_switch = inp == 'O'

        self.DICOM_IDX = -1
        next(self)

    def __next__(self):

        self.DICOM_IDX += 1
        print('List index:', self.DICOM_IDX)
        if self.DICOM_IDX <= self.n_dicom_files - 1:
            self.raw_dicom_path = self.raw_dicom_paths[self.DICOM_IDX]
            self.clean_dicom_path = self.clean_data_dp_ + '/'.join(self.raw_dicom_path.split('/')[self.clean_data_dp_last_dir_idx:])
            print('Raw DICOM file path:', self.raw_dicom_path)
            print('Clean DICOM file path:', self.clean_dicom_path)
            if os.path.exists(self.clean_dicom_path):
                print('W: DICOM file already exists at the output path\n%s'%(self.clean_dicom_path))
                if self.overwrite_switch:
                    print('W: Duplicate file will potentially be overwritten')
                else:
                    print('Ignoring\n---', end = 2 * '\n')
                    next(self)
        else:
            self.raw_dicom_path = None
            self.clean_dicom_path = None

        ## Print input path and output path

    def copy_dir_structure(self):
        '''
            Description:
                Generates an empty replica of the input directory. The replica is placed inside '../dataset/clean/'. It also includes all non-DICOM files.
        '''

        def copy_dir_structure_():
            '''
                Description:
                    It recursively parses all directory paths and copies structure to the clean directory. Output directory must not exist prior to this.
            '''

            shutil.copytree\
            (
                src = self.raw_data_dp,
                dst = self.clean_data_dp,
                ignore = ffilter
            )
            print('Created a new output directory', end = 2 * '\n')

        ## Rule where for an existing filesystem path, if the path corresponds to a DICOM file path it is added to the output's list.
        def ffilter(dir, all_files):
            filtered_files = []
            for f in all_files:
                if ( f.split('.')[-1] in self.modifiable_file_extension_names ) and ( os.path.isfile(os.path.join(dir, f)) ):
                    filtered_files.append(f)

            return filtered_files

        if os.path.exists(self.clean_data_dp):
            print('W: Output directory already exists. Attempting to delete it.')
            self.rm_out_dir()

            ## This can only happen if self.SAFETY_SWITCH is set to True
            if os.path.exists(self.clean_data_dp):
                print('W: Failed to delete output directory.')
                inp_choices = {'A': 'Abort', 'D': 'Delete', 'I': 'Ignore'}
                inp = input('\nAvailable choices:\n[A] Abort operation\n[D] Override output directory deletion and make a fresh copy of the input directory structure\n[I] Ignore; select this only if you are sure that the output directory structure is a replica of the input directory structure\n> ')

                while inp not in inp_choices.keys():
                    inp = input('Select a valid input:\n> ')

                print('Proceeding with:', inp_choices[inp])
                if inp == 'D':
                    ## The previous if can only happen if self.SAFETY_SWITCH is set to True
                    self.SAFETY_SWITCH = False
                    self.rm_out_dir()
                    self.SAFETY_SWITCH = True
                    copy_dir_structure_()
                elif inp == 'I':
                    pass
                else:
                    exit()
            else:
                copy_dir_structure_()
        else:
            copy_dir_structure_()

    def rm_out_dir(self):
        '''
            Description:
                This removes the output directory.
        '''

        if self.SAFETY_SWITCH:
            print('W: Safety switch is on, hence output directory will not be deleted.')
        else:
            shutil.rmtree(self.clean_data_dp)
            print('W: Output directory deleted.')

    def generate_dicom_paths(self, data_dp):

        dicom_paths = []
        for extension_name in self.modifiable_file_extension_names:
            dicom_paths += \
            (
                glob\
                (
                    pathname = data_dp + '**/*.' + extension_name,
                    recursive = True
                )
            )

        return dicom_paths

    def parse_file(self):

        dcm = dicom.dcmread(self.raw_dicom_path)

        return dcm

    def export_processed_file(self, dcm):

        dcm.save_as(self.clean_dicom_path)

    def store_fig(self, figure):

        fp = '.'.join(self.clean_dicom_path.split('.')[:-1]) + '.png'
        plt.savefig(fp, dpi = 1200)
        plt.close()
        figure.clear()