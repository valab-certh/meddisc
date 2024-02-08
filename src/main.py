import os, logging

## Warning supression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

## Warning supression
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
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
import base64
from segment_anything import sam_model_registry
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tiny_vit_sam import TinyViT
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

## SegSAM Dependencies
import torch

class user_options_class(BaseModel):
    ## These have to match exactly with javascript's "dictionary" keys, both the keys and the data types
    clean_image: bool
    retain_safe_private: bool
    retain_uids: bool
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

class ResponseModel(BaseModel):
    message: str

class DicomData(BaseModel):
    pixelData: str
    filepath: str

class BoxData(BaseModel):
    normalizedStart: Dict
    normalizedEnd: Dict
    segClass: str
    inpIdx: int

def clean_all():

    clean_config_session()
    clean_imgs()

def clean_config_session():

    if os.path.isfile('./session_data/user_input.json'):
        os.remove('./session_data/user_input.json')

    if os.path.isfile('./session_data/session.json'):
        os.remove('./session_data/session.json')

    if os.path.isfile('./session_data/requested_action_group_dcm.csv'):
        os.remove('./session_data/requested_action_group_dcm.csv')

    if os.path.isfile('./session_data/custom_config.csv'):
        os.remove('./session_data/custom_config.csv')

def clean_imgs():

    dp, _, fps = list(os.walk('./session_data/raw'))[0]
    for fp in fps:
        if fp != '.gitkeep':
            os.remove(dp + '/' + fp)

    if os.path.exists('./session_data/clean/de-identified-files'):
        shutil.rmtree('./session_data/clean/de-identified-files')

    dp, _, fps = list(os.walk('./static/client_data'))[0]
    for fp in fps:
        if fp != '.gitkeep':
            os.remove(dp + '/' + fp)

def DCM2DictMetadata(ds: pydicom.dataset.Dataset) -> pydicom.dataset.Dataset:

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

    ## Resets directories
    clean_all()

    return FileResponse('./static/index.html')

@app.post('/conversion_info')
async def conversion_info(dicom_pair_fp: List[str] = Body(...)):

    raw_dcm = pydicom.dcmread(dicom_pair_fp[0])
    cleaned_dcm = pydicom.dcmread(dicom_pair_fp[1])
    downscale_dimensionality = 1024

    raw_img = keras_ocr_preprocessing(raw_dcm.pixel_array, downscale_dimensionality = downscale_dimensionality, multichannel = True)
    raw_hash = hashlib.sha256(raw_img.tobytes()).hexdigest()
    raw_img_fp = './static/client_data/' + raw_hash + '.png'
    Image.fromarray(raw_img).save(raw_img_fp)

    cleaned_img = keras_ocr_preprocessing(cleaned_dcm.pixel_array, downscale_dimensionality = downscale_dimensionality, multichannel = True)
    cleaned_hash = hashlib.sha256(cleaned_img.tobytes()).hexdigest()
    cleaned_img_fp = './static/client_data/' + cleaned_hash + '.png'
    Image.fromarray(cleaned_img).save(cleaned_img_fp)

    # with open(file = '../raw_dcm_meta.json', mode = 'w') as file:
    #     json.dump(DCM2DictMetadata(ds = raw_dcm), fp = file)

    # with open(file = '../cleaned_dcm_meta.json', mode = 'w') as file:
    #     json.dump(DCM2DictMetadata(ds = cleaned_dcm), fp = file)

    try:
        mask = cleaned_dcm.SegmentSequence[0].PixelData
    except:
        mask = np.zeros(shape = (cleaned_dcm.Rows, cleaned_dcm.Columns), dtype = np.uint8)

    return \
    {
        'raw_dicom_metadata': DCM2DictMetadata(ds = raw_dcm),
        'raw_dicom_img_fp': raw_img_fp,
        'cleaned_dicom_metadata': DCM2DictMetadata(ds = cleaned_dcm),
        'cleaned_dicom_img_fp': cleaned_img_fp,
        'segmentation_data': base64.b64encode(mask).decode('utf-8'),
        'dimensions': [cleaned_dcm.Rows, cleaned_dcm.Columns]
    }

@app.post('/reset_mask/')
async def reset_mask(current_dcm_fp: str = Body(...)):
    current_dcm = pydicom.dcmread(current_dcm_fp)
    return \
    {
        'PixelData': base64.b64encode(current_dcm.SegmentSequence[0].PixelData).decode('utf-8'),
        'dimensions': [current_dcm.Rows, current_dcm.Columns]
    }

@app.post('/modify_dicom/')
async def modify_dicom(data: DicomData):
    pixelData = base64.b64decode(data.pixelData)
    filepath = data.filepath
    modified_dcm = pydicom.dcmread(filepath)
    modified_dcm.SegmentSequence[0].PixelData = pixelData
    modified_dcm.save_as(filepath)
    return \
    {
        'success': True
    }

@app.post('/upload_files/')
async def get_files(files: List[UploadFile] = File(...)):

    clean_imgs()

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
            total_uploaded_file_bytes += len(contents)
        except InvalidDicomError:
            print('W: The following path does not correspond to a DICOM file\n%s'%(fp))
            os.remove(fp)
            print('Irrelevant file deleted')

    total_uploaded_file_megabytes = '%.1f'%(total_uploaded_file_bytes / (10**3)**2)

    return {'n_uploaded_files': len(proper_dicom_paths), 'total_size': total_uploaded_file_megabytes}

@app.post('/correct_segmentation_sequence')
async def correct_segmentation_sequence():

    with open(file = './session_data/user_options.json', mode = 'r') as file:
        user_input = json.load(file)

    fps = glob(os.path.join(user_input['input_dcm_dp'], '*'))

    for fp in fps:

        classes_idx2name = ['lesion', 'lung']

        dcm = pydicom.dcmread(fp)

        ## Improvement 0x8fee8c92e9 -> This should be replaced by our own IOD
        if (0x0062, 0x0002) in dcm and dcm.SegmentSequence[0].SegmentDescription == ';'.join(classes_idx2name):
            continue
        else:
            img_shape = dcm.pixel_array.shape
            mask = np.zeros(shape = img_shape, dtype = np.uint8)
            dcm = attach_segm_data(dcm = dcm, seg_mask = mask, class_names = classes_idx2name)

        dcm.save_as(fp)

@app.post('/session')
async def handle_session_button_click(session_dict: Dict[str, Any]):
    with open(file = './session_data/session.json', mode = 'w') as file:
        json.dump(session_dict, file)

@app.post("/custom_config/")
async def get_files(ConfigFile: UploadFile = File(...)):
    contents = await ConfigFile.read()
    with open(file = './session_data/custom_config.csv', mode = 'wb') as file:
        file.write(contents)

@app.post("/medsam_estimation/")
async def medsam_estimation(boxdata: BoxData):
    ## Currently works for exactly 1 bounding box

    start = boxdata.normalizedStart
    end = boxdata.normalizedEnd
    segClass = boxdata.segClass
    inpIdx = boxdata.inpIdx
    # process data here

    bbox = np.array([min(start['x'],end['x']), min(start['y'],end['y']), max(end['x'],start['x']), max(end['y'], start['y'])])

    # transfer box_np t0 1024x1024 scale
    box_256 = bbox[None, :] * 256

    print('Starting segmentation')
    t0 = time.time()
    medsam_seg = medsam_inference(medsam_model, embeddings[inpIdx], box_256, (newh, neww), (Hs[inpIdx], Ws[inpIdx]))
    print('Segmentation completed in %.2f seconds'%(time.time()-t0))

    return \
    {
        'mask': base64.b64encode(medsam_seg).decode('utf-8'),
        'dimensions': [Ws[inpIdx], Hs[inpIdx]]
    }

@app.post('/submit_button')
async def handle_submit_button_click(user_options: user_options_class):
    '''
        Description:
            De-identification submit button handler.

        Args:
            user_options. User's configuration for the de-identification process from the predefined interfaces options. This does not refer, for example to the custom user's configuration provided via a .csv file.

        Returns:
            dicom_pair_fps. Contains pairs of paths. Each pair consists of a source (raw) DICOM file path and a target (cleaned) DICOM file path.
    '''

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

    ## MedSAM initialization
    prepare_medsam()

    ## Masks initialization
    initialize_masks()

    return dicom_pair_fps

def initialize_masks():

    global seg_masks
    seg_masks = []

    global classes
    classes = []

    for inpIdx in range(len(embeddings)):
        seg_masks.append(np.zeros(shape = (Hs[idx], Ws[idx]), dtype = np.uint8) for idx in range(len(embeddings)))

class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device='cpu')
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

def prepare_medsam():
    '''
        Description:
            For each of the raw DICOM images, this function is responsible for deserializing a MedSAM checkpoint. It also applies a computationally intensive transformation of input images that converts raw input to their corresponding embeddings (their MedSAM encodings).
    '''

    global embeddings
    embeddings = []

    global Hs, Ws
    Hs, Ws = [], []

    global medsam_model
    global newh, neww

    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64, ## (64, 256, 256)
            128, ## (128, 128, 128)
            160, ## (160, 64, 64)
            320 ## (320, 64, 64) 
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
    )
    # %%
    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )

    medsam_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )
    medsam_lite_checkpoint = torch.load('./pretrained_segmenters/MedSAM/lite_medsam.pth', map_location='cpu')
    medsam_model.load_state_dict(medsam_lite_checkpoint)
    medsam_model.to('cpu')

    print('MedSAM model deserialization completed')

    dcm_fps = sorted(glob('./session_data/raw/*'))

    t0 = time.time()
    print('Initializing MedSAM embeddings')

    for dcm_fp in dcm_fps:

        img = pydicom.dcmread(dcm_fp).pixel_array

        ## Input Preprocessing
        if len(img.shape) == 2:
            img_3c = np.repeat(img[:, :, None], 3, axis=-1)
        else:
            img_3c = img
        H, W, _ = img_3c.shape
        Hs.append(H)
        Ws.append(W)

        # image preprocessing
        img_256 = cv2.resize(src = img_3c, dsize = (256, 256)).astype(np.float32)
        newh, neww = img_256.shape[:2]
        img_256 = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_256_tensor = (
            torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0)#.to(device)
        )

        with torch.no_grad():
            embeddings.append(medsam_model.image_encoder(img_256_tensor))  # (1, 256, 64, 64)

    print('Initialization completed - %.2f'%(time.time()-t0))

def dicom_deidentifier(SESSION_FP: None or str = None) -> tuple[dict, list[tuple[str]]]:
    '''
        Description:
            Applies de-identification to a set of DICOM files based on the configuration file `user_options.json`. Additionally it can consider a session file, namely `session.json` in case a session was interrupted.

            Warning:
                Session files are not supposed to be modified between interruptions, neither their corresponding DICOM files.

        Args:
            SESSION_FP. File path for the session.json file. If `SESSION_FP` is set to None then a new session file will be created at the parent directory.

        Returns:
            session. This dictionary contains the session file's content.
                {
                    `Patient_ID0`:
                    {
                        'patientPseudoId': '000000',
                        'daysOffset': 7172,
                        'secondsOffset': 34283,
                        ...
                    },
                    `Patient_ID1`:
                    {
                        'patientPseudoId': '000001',
                        'daysOffset': 2230,
                        'secondsOffset': 14928,
                        ...
                    },
                    ...
                }
            path_pairs. Contains raw-cleaned DICOM pairs corresponding to the input.
    '''

    ## Set to True if you want to invoke NVIDIA GPU
    GPU = True

    if (not GPU):
        tf.config.set_visible_devices([], 'GPU')
        print('[DISABLED] PARALLEL COMPUTATION\n\n---')
    elif len(tf.config.list_physical_devices('GPU')) == 0:
        print('W: No GPU detected, switching to CPU instead')
        print('[DISABLED] PARALLEL COMPUTATION\n\n---')
    elif tf.config.list_physical_devices('GPU')[0][1] == 'GPU':
        print('[ENABLED] PARALLEL COMPUTATION\n\n---')

    ## Get the custom options from user
    if os.path.isfile('./session_data/custom_config.csv'):
        custom_config_df = pd.read_csv(filepath_or_buffer = './session_data/custom_config.csv', index_col = 0)

        # Strip single quotes from Tag IDs in the Custom table
        custom_config_df.index = custom_config_df.index.str.strip("'")
    else:
        custom_config_df = None

    ## Parse all predefined DICOM metadata configurations
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

    ## Define metadata action group based on user input
    requested_action_group_df = get_action_group(user_input = user_input, action_groups_df = action_groups_df, custom_config_df = custom_config_df)

    ## Store the user's action group
    requested_action_group_df.to_csv('./session_data/requested_action_group_dcm.csv')

    ## Get iterator for input DICOM files
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

        ## Adjusts DICOM metadata based on user parameterization
        dcm, tag_value_replacements = adjust_dicom_metadata\
        (
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
            dcm = image_deintentifier(dcm = dcm)

        print('DICOM Processing Completed')

        ## Store DICOM file and create output directories
        rw_obj.export_processed_file(dcm = dcm)

        ## Overwrite session file to save progress
        rw_obj.export_session(session = session)

    print('Operation completed')

    return session, rw_obj.dicom_pair_fps

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):

    box_torch = torch.as_tensor(box_256, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg

def seg_est_covid19(dcm: pydicom.dataset.FileDataset) -> tuple[np.ndarray, list[str]]:
    '''
        Description:
            Responsible for estimating the segmentation mask for the (1) lesions caused by the Covid-19 virus along with (2) lung areas. In the current version it takes into account exactly 1 example per run.

        Args:
            dcm. DICOM file to be predicted.

        Returns:
            mask. Shape (H, W). Segmentation mask. Each pointed element takes 8 bits.
            class_names. Contains an ordered set of classes.
    '''

    img = dcm.pixel_array
    H, W = img.shape

    model = load_model('./pretrained_segmenters/covid19')

    ## ! Preprocess: Begin

    img = img.astype(np.float32) / 4095

    ## Resize
    img = cv2.resize(src = img, dsize = (256, 256)).astype(np.float32)

    ## Add batch axis
    img = np.expand_dims(img, axis=0)

    ## Add channel axis
    img = np.expand_dims(img, axis=-1)

    ## ! Preprocess: End

    seg_probs = model.predict(img)[0, ...]

    ## Postprocess
    mask = cv2.resize(np.argmax(seg_probs, axis = -1), dsize = (W, H), interpolation = cv2.INTER_NEAREST).astype(np.uint8)

    return mask, ['lesion', 'lung']

def attach_segm_data(dcm: pydicom.dataset.FileDataset, seg_mask: np.array, class_names: list[str]) -> pydicom.dataset.FileDataset:
    '''
        Description:
            Attaches necessary segmentation data in the header's DICOM structure.

        Args:
            dcm. Input DICOM file's content.
            seg_mask. Shape (H, W). Segmentation mask. Each pointed element holds unsigned integers with size 8 bits.
            class_names. Contains all the classes names in order, excluding the background class.

        Returns:
            dcm_. Contains the segmentation sequence attribute, structured as follows

            (0062,0002) SQ SegmentSequence = <sequence of 1 item>
            #0
                (0008,0016) UI SOPClassUID = 1.2.840.10008.5.1.4.1.1.66.4
                (0028,0010) US Rows = N_ROWS
                (0028,0011) US Columns = N_COLS
                (0028,0100) US BitsAllocated = 8
                (0062,0006) ST SegmentDescription = CLASS_1_NAME;...CLASS_N_NAME
                (7FE0,0010) OB PixelData = <binary data of length: 786432>

            where N_ROWS and N_COLS are the image's number of rows and columns respectively. Also CLASS_1_NAME;...CLASS_N_NAME is the correctly ordered sequence of foreground class names.

            The internal PixelData must hold a serialization of a numpy array with shape (H, W) which is a segmentation mask.
    '''

    assert type(seg_mask[0, 0]) == np.uint8, 'E: Incompatible element-wise data type'

    seg_dataset = pydicom.dataset.Dataset()

    img = dcm.pixel_array

    assert len(img.shape) == 2, 'E: Incompatible image shape'

    seg_dataset.Rows, seg_dataset.Columns = img.shape
    seg_dataset.SOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'
    seg_dataset.BitsAllocated = 8
    seg_dataset.SegmentDescription = ';'.join(class_names)
    seg_dataset.PixelData = seg_mask.tobytes()

    dcm.SegmentSequence = pydicom.sequence.Sequence([seg_dataset])

    return dcm

def deidentification_attributes(user_input: dict, dcm: pydicom.dataset.FileDataset) -> pydicom.dataset.FileDataset:
    '''
        Description:
            Appends additional DICOM attributes that specify the anonymization process as per Table CID 7050 from https://dicom.nema.org/medical/dicom/2019a/output/chtml/part16/sect_CID_7050.html. Some of these additional attributes are specified at https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_E.3.html.

        Args:
            user_input. Identical to the content of `user_options.json`.
            dcm. Raw DICOM input.

        Returns:
            dcm_. DICOM object where de-identification tags were added.
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

def keras_ocr_preprocessing(img: np.ndarray, downscale_dimensionality: int, multichannel: bool = True) -> np.ndarray:
    '''
        Description:
            Image preprocessing pipeline. It is imperative that the image is converted to (1) uint8 and in (2) RGB in order for keras_ocr's detector to properly function.

        Args:
            img. Shape (H, W). Monochrome input image.
            downscale_dimensionality. Downscale H and W of output image. If set to 0, False or None it does not apply downscaleing in the input image.
            multichannel. If set to True then appends an additional dimension and sets it as the color channel.

        Returns:
            out_image. Preprocessed image where each element has `np.uint8` data type. Each pixel is normalized. Its shape is (H, W) if `multichannel` is set to False, otherwise its shape is (H, W, 3).
    '''

    img = (255.0 * ((img - np.min(img)) / (np.max(img) - np.min(img)))).astype(np.uint8)

    if downscale_dimensionality:
        new_shape = (min([downscale_dimensionality, img.shape[0]]), min([downscale_dimensionality, img.shape[1]]))
        img = cv2.resize(img, (new_shape[1], new_shape[0]))

    if (multichannel) and (len(img.shape) == 2):
        img = np.stack(3*[img], axis = -1)

    return img

def bbox_area_distorter(img: np.ndarray, bboxes: np.ndarray, initial_array_shape: tuple[int], downscaled_array_shape: tuple[int]) -> np.ndarray:
    '''
        Description:
            Redacts image area corresponding to bounding boxes. Applied for the de-identification of burned-in text in images.

        Args:
            img. Shape (H, W). Input image.
            bboxes. Shape (n_bboxes, 4, 2). Includes all the estimated bounding boxes from `img`. The 4 in the shape is the number of vertices for each box and 2 are the plane coordinates. The vertices inside the bboxes array should be sorted in a way that corresponds to a geometrically counter-clockwise order. For example given a non-rotated (0 degree) bounding box with index 0, the following rule applies
                bboxes[0, 0, :] -> upper left vertex
                bboxes[0, 1, :] -> lower left vertex
                bboxes[0, 2, :] -> lower right vertex
                bboxes[0, 3, :] -> upper right vertex
            initial_array_shape. The shape of the raw input image. Under normal circumstances, this shape is equal to `img.shape`.
            downscaled_array_shape. The shape of the downscaled input image.

        Returns:
            img_. Input image where all its corresponding bounding box areas were redacted.
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
            dtype = np.int32 ## Must remain this way, otherwise cv2.fillPoly will throw an error
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

def image_deintentifier(dcm: pydicom.dataset.FileDataset) -> pydicom.dataset.FileDataset:
    '''
        Description:
            Removes detected pixel data burned-in text from input DICOM image.

        Args:
            dcm.

        Returns:
            dcm_.
    '''

    min_dim = 50
    downscale_dimensionality = 1024

    assert downscale_dimensionality >= min_dim, 'E: Downscale dimensionality is excessively small'

    ## Extract image data from dicom files
    ## Scalar data type -> uint16
    dcm.decompress()
    raw_img_uint16_grayscale = dcm.pixel_array

    if min(raw_img_uint16_grayscale.shape) < min_dim:

        print('W: Pixel data will not be affected because the DICOM image resolution is excessively small')

        return dcm

    ## Secondary information about the DICOM file
    print('Input DICOM file information')
    print('Input image shape: ', raw_img_uint16_grayscale.shape)

    if downscale_dimensionality < max(raw_img_uint16_grayscale.shape[0], raw_img_uint16_grayscale.shape[1]):
        print('Downscaling detection input image from shape (%d, %d) to (%d, %d)'%(raw_img_uint16_grayscale.shape[0], raw_img_uint16_grayscale.shape[1], downscale_dimensionality, downscale_dimensionality))
    raw_img_uint8_rgb = keras_ocr_preprocessing(img = raw_img_uint16_grayscale, downscale_dimensionality = downscale_dimensionality)

    pipeline = keras_ocr.detection.Detector()

    ## Returns a ndarray with shape (n_bboxes, 4, 2) where 4 is the number of points for each box, 2 are the plane coordinates.
    bboxes = pipeline.detect([raw_img_uint8_rgb])[0]

    initial_array_shape = raw_img_uint16_grayscale.shape
    downscaled_array_shape = raw_img_uint8_rgb.shape[:-1]

    if np.size(bboxes) != 0:

        cleaned_img = bbox_area_distorter\
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

    return dcm

def get_action_group(user_input: dict, action_groups_df: pd.core.frame.DataFrame, custom_config_df: pd.core.frame.DataFrame or None) -> pd.core.frame.DataFrame:
    '''
        Description:
            Depending on the user's choice, and an action group lookup table, based on Nema's action principles as per Table E.1-1a and Table E.1-1 of https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_e.html, an action group is generated. That action group is in the form of a column from Table E.1-1 acting as a configuration basis for the anonymization of a DICOM file.

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
            custom_config_df. Contains two columns, namely "Tag ID" and "Action". The actions will be absorbed by the requested action group.

        Returns:
            requested_action_group_df. Requested action group, defined by the user's choice. Each row corresponds to an attribute. The index column contains tag codes. Column 0 contains the tag names. Column 2 contains the action code.
    '''

    def merge_action(primary_srs: pd.core.series.Series, Action2BeAssigned_srs: pd.core.series.Series) -> pd.core.series.Series:
        '''
            Description:
                Overwrites tag actions that already exist in rows of Action2BeAssigned_srs. The rows that do not include an action (the latter ones are set to NaN in `Action2BeAssigned_srs`) are omitted.

            Args:
                primary_srs. Contains tag actions from a predefined range of de-identification actions.
                Action2BeAssigned_srs. All tag values that are intented to be replaced in the corresponding rows of `primary_srs`.

            Returns:
                primary_srs_. `primary_srs` where each row value was substituted with a (non-NaN) value from `Action2BeAssigned_srs` from its corresponding tag position.
        '''

        return primary_srs.where\
        (
            cond = Action2BeAssigned_srs.isna(),
            other = Action2BeAssigned_srs,
            axis = 0,
            inplace = False,
        )

    def merge_with_custom_user_config_file(requested_action_group_df, custom_config_df):

        ## Check if 'Action' values are valid
        valid_actions = {'X', 'K', 'C'}
        if not set(custom_config_df['Action']).issubset(valid_actions):
            print('E: "Action" values in the Custom table must be either "X", "K", or "C". Please correct the data.')
            exit()

        requested_action_group_df = requested_action_group_df.merge(custom_config_df[['Action']], left_index=True, right_index=True, how='left')
        requested_action_group_df.loc[requested_action_group_df['Action'].isin(['X', 'K', 'C']), 'Requested Action Group'] = requested_action_group_df['Action']
        requested_action_group_df.drop(columns=['Action'], inplace=True)

        return requested_action_group_df


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

        requested_action_group_df['Requested Action Group'] = merge_action(primary_srs = requested_action_group_df['Requested Action Group'], Action2BeAssigned_srs = action_groups_df['Rtn. Safe Priv. Opt.'])

    if user_input['retain_uids']:

        requested_action_group_df['Requested Action Group'] = merge_action(primary_srs = requested_action_group_df['Requested Action Group'], Action2BeAssigned_srs = action_groups_df['Rtn. UIDs Opt.'])

    if user_input['retain_device_identity']:

        requested_action_group_df['Requested Action Group'] = merge_action(primary_srs = requested_action_group_df['Requested Action Group'], Action2BeAssigned_srs = action_groups_df['Rtn. Dev. Id. Opt.'])

    if user_input['retain_patient_characteristics']:

        requested_action_group_df['Requested Action Group'] = merge_action(primary_srs = requested_action_group_df['Requested Action Group'], Action2BeAssigned_srs = action_groups_df['Rtn. Pat. Chars. Opt.'])

    if user_input['date_processing'] == 'keep':

        requested_action_group_df['Requested Action Group'] = merge_action(primary_srs = requested_action_group_df['Requested Action Group'], Action2BeAssigned_srs = action_groups_df['Rtn. Long. Modif. Dates Opt.'])

    elif user_input['date_processing'] == 'offset':

        requested_action_group_df['Requested Action Group'] = merge_action(primary_srs = requested_action_group_df['Requested Action Group'], Action2BeAssigned_srs = action_groups_df['Offset Long. Modif. Dates Opt.'])

    elif user_input['date_processing'] == 'remove':

        requested_action_group_df['Requested Action Group'] = merge_action(primary_srs = requested_action_group_df['Requested Action Group'], Action2BeAssigned_srs = action_groups_df['Remove Long. Modif. Dates Opt.'])

    if user_input['retain_descriptors']:

        requested_action_group_df['Requested Action Group'] = merge_action(primary_srs = requested_action_group_df['Requested Action Group'], Action2BeAssigned_srs = action_groups_df['Rtn. Desc. Opt.'])

    if type(custom_config_df) == pd.core.frame.DataFrame:
        requested_action_group_df = merge_with_custom_user_config_file(requested_action_group_df, custom_config_df)

    return requested_action_group_df

def adjust_dicom_metadata(dcm: pydicom.dataset.FileDataset, action_group_fp: str, patient_pseudo_id: str, days_total_offset: int, seconds_total_offset: int) -> (pydicom.dataset.FileDataset, dict):
    '''
        Description:
            Peforms metadata de-identification on a DICOM object based on a configuration file.

        Args:
            dcm. Input DICOM object.
            action_group_fp. Path for .csv configuration file that contains the an action group. The file's content is saved in action_group_df.
                action_group_df. Specifies how exactly the DICOM's metadata will be modified according to its attribute actions.
            patient_pseudo_id. The patient's ID and name will be replaced by this value.
            days_total_offset. If the date values are set to clean (C action) then they will be replaced by the sum of their already existing ones, with this offset. The offset is a value sampled from the uniform distribution.
            seconds_total_offset.

        Returns:
            updated_dcm. DICOM object adjusted according to configuration.
            tag_value_replacements. Contains things like patient pseudo ID and date offset.
    '''

    def add_date_offset(input_date_str: str, days_total_offset: str) -> str:
        '''
            Description:
                Adds a pseudo date offset to input date value.

            Args:
                input_date_str. Input date in YYYYMMDD format.
                days_total_offset. Date offset in days.

            Returns:
                output_date_str. Resulting date in YYYYMMDD format.
        '''

        input_date = datetime.datetime.strptime(input_date_str, '%Y%m%d')
        output_date = input_date + datetime.timedelta(days = days_total_offset)
        output_date_str = output_date.strftime('%Y%m%d')

        return output_date_str

    def seconds2daytime(seconds_total_offset: int) -> str:
        '''
            Description:
                Converts day time seconds to day time in a proper format.

            Args:
                seconds_total_offset. Day time in seconds.

            Returns:
                output_time_str. Day time in hhmmss format.
        '''

        output_hours = seconds_total_offset // 3600
        output_minutes = (seconds_total_offset % 3600) // 60
        output_seconds = (seconds_total_offset % 3600) % 60

        output_time_str = '%.2d%.2d%.2d'%(output_hours, output_minutes, output_seconds)

        return output_time_str

    def recursive_SQ_cleaner(ds: pydicom.dataset.Dataset or pydicom.dataset.FileDataset, action: str, action_attr_tag_idx: str) -> pydicom.dataset.Dataset or pydicom.dataset.FileDataset:
        '''
            Description:
                Cleans DICOM file's metadata based on one de-identification action and it does so recursively with respect to sequence depth.

            Args:
                ds. For a given DICOM sequence, this contains all dataset tags that will be cleaned. If a SQ type VR exists among the tags, then it is handled by a loop where for each of its containing dataset the same function is applied.

            Returns:
                ds_. A dataset object's that was cleaned.
        '''

        for ds_attr in ds:
            ds_tag_idx = re.sub('[(,) ]', '', str(ds_attr.tag))
            if ds[ds_tag_idx].VR == 'SQ':
                for inner_ds_idx in range(ds[ds_tag_idx].VM):
                    ds[ds_tag_idx].value[inner_ds_idx] = recursive_SQ_cleaner\
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

                    if ds[ds_tag_idx].value != '' and ds[ds_tag_idx].VR == 'DA':

                        tag_value_replacements['days_total_offset'] = days_total_offset

                        ## The offset is constant among DICOM files corresponding to the same patient
                        ds[ds_tag_idx].value = add_date_offset(input_date_str = ds[ds_tag_idx].value, days_total_offset = days_total_offset)

                    ## (!) The current implementation of TM VR pseudonymization replaces the timestamp with a common random value, rather than using an offset. This approach is simpler and avoids potential issues with 24-hour overflows, which could inadvertently increment the date by one day.
                    elif ds[ds_tag_idx].VR == 'TM':

                        tag_value_replacements['seconds_total_offset'] = seconds_total_offset

                        ds[ds_tag_idx].value = seconds2daytime(seconds_total_offset = tag_value_replacements['seconds_total_offset'])

        return ds

    action_group_df = pd.read_csv(filepath_or_buffer = action_group_fp, index_col = 0)

    tag_value_replacements = dict()

    ## Will be replaced only if date and time offsets are applied to at least one tag
    tag_value_replacements['days_total_offset'] = 0
    tag_value_replacements['seconds_total_offset'] = 0

    ## Cleaning primary DICOM object
    ## Scans all DICOM tags until it intercepts `action_attr_tag_idx`. Then it updates that exact tag index per dataset. You can view this recursion statically, as an arbitrary directed graph.
    ## Each leaf node counts as one attribute update. Collectively this can alter multiple tags of the DICOM object.
    for action_attr_tag_idx in action_group_df.index:
        action = action_group_df.loc[action_attr_tag_idx].iloc[1]
        dcm = recursive_SQ_cleaner\
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
        '''
            Args:
                in_dp. Source directory path of raw DICOM files.
                out_dp. Target directory path where the cleaned DICOM files will be exported at.
        '''

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

    def __next__(self) -> bool:

        self.DICOM_IDX += 1
        if self.DICOM_IDX <= self.n_dicom_files - 1:
            self.raw_dicom_path = self.raw_dicom_paths[self.DICOM_IDX]
            print('---\n')
            print('DICOM List Index:', self.DICOM_IDX)
            return True
        else:
            return False

    def get_dicom_paths(self, data_dp: str) -> list:
        '''
            Description:
                Gets file path list of all DICOM files inside a given directory, independently of file extension.

            Args:
                data_dp. Directory path containing DICOM files.

            Returns:
                proper_dicom_paths. Contains all file subsequent DICOM file paths of `data_dp`.
        '''

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

    def parse_file(self) -> pydicom.dataset.FileDataset or False:
        '''
            Description:
                Reads a DICOM file only if it was not converted by the current session. The comparison is made based on the raw files' hashes.

            Returns:
                dcm. The parsed file's DICOM object or if the same file was already converted, it returns False.
        '''

        self.input_dicom_hash = hashlib.sha256(self.raw_dicom_path.encode('UTF-8')).hexdigest()

        if self.input_dicom_hash in self.hashes_of_already_converted_files:
            return False
        else:
            dcm = pydicom.dcmread(self.raw_dicom_path)
            print('Parsed\n%s'%(self.raw_dicom_path))
            return dcm

    def export_processed_file(self, dcm: pydicom.dataset.FileDataset):
        '''
            Description:
                Exports processed DICOM object to a DICOM file with its filename set as its corresponding input file's hash, with "dcm" file extension.

            Args:
                dcm. Processed DICOM object intended for exportation.
        '''

        self.clean_dicom_dp = self.clean_data_dp + str(dcm[0x0010, 0x0020].value) + '/' + str(dcm[0x0008, 0x0060].value) + '/' + str(dcm[0x0020, 0x0011].value)
        if not os.path.exists(self.clean_dicom_dp):
            os.makedirs(self.clean_dicom_dp)

        clean_dicom_fp = self.clean_dicom_dp + '/' + self.input_dicom_hash + '.dcm'

        print('Exporting file at\n%s'%(clean_dicom_fp))

        dcm.save_as(clean_dicom_fp)

        self.dicom_pair_fps.append((self.raw_dicom_path, clean_dicom_fp))

    def export_session(self, session: dict):
        '''
            Description:
                Exports session object on a predetermined file path.

            Args:
                session. The session object.
        '''

        with open(self.clean_data_dp + '/session.json', 'w') as file:
            json.dump(session, file)