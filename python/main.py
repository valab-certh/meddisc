#!/usr/bin/env python3
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sys
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import keras_ocr
import numpy as np
import pandas as pd
import pydicom
import torch
from fastapi import Body, FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel
from pydicom.errors import InvalidDicomError
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from torch import nn
from torch.nn import functional
from uvicorn import run

if TYPE_CHECKING:
    from numpy.typing import NDArray


class UserOptionsClass(BaseModel):
    clean_image: bool
    retain_safe_private: bool
    retain_uids: bool
    retain_device_identity: bool
    retain_patient_characteristics: bool
    date_processing: str
    retain_descriptors: bool
    patient_pseudo_id_prefix: str


class SessionPatientInstanceClass(BaseModel):
    patient_pseudo_id: str
    days_offset: float
    seconds_offset: int


class SessionClass(BaseModel):
    patients: dict[str, SessionPatientInstanceClass]


class ResponseModel(BaseModel):
    message: str


class DicomData(BaseModel):
    pixel_data: str
    filepath: str
    classes: list[str]


class BoxData(BaseModel):
    normalized_start: dict[str, float]
    normalized_end: dict[str, float]
    seg_class: int
    inp_idx: int


class BoxDataResponse(BaseModel):
    mask: str
    dimensions: list[int]


class ConversionInfoResponse(BaseModel):
    raw_dicom_metadata: dict[str, dict[str, str]]
    raw_dicom_img_data: str
    cleaned_dicom_metadata: dict[str, dict[str, str]]
    cleaned_dicom_img_data: str


class MaskFromFileResponse(BaseModel):
    PixelData: str
    dimensions: list[int]


class ModifyResponse(BaseModel):
    success: bool


class UploadFilesResponse(BaseModel):
    n_uploaded_files: int
    total_size: str


def clean_config_session() -> None:
    session_fp = Path("./tmp/session-data/session.json")
    if session_fp.is_file():
        session_fp.unlink()
    action_fp = Path("./tmp/session-data/requested-action-group-dcm.csv")
    if action_fp.is_file():
        action_fp.unlink()
    custom_fp = Path("./tmp/session-data/custom-config.csv")
    if custom_fp.is_file():
        custom_fp.unlink()


def clean_imgs() -> None:
    dp, _, fps = next(iter(os.walk("./tmp/session-data/raw")))
    for fp in fps:
        if fp != ".gitkeep":
            Path(dp + "/" + fp).unlink()
    fps = list(Path("./tmp/session-data/clean").glob("*"))
    for fp in fps:
        if str(fp).split(".")[-1] == "png":
            Path(fp).unlink()
    if Path("./tmp/session-data/clean/de-identified-files").exists():
        shutil.rmtree("./tmp/session-data/clean/de-identified-files")


def clean_all() -> None:
    clean_config_session()
    clean_imgs()


def dcm2dictmetadata(ds: pydicom.dataset.Dataset) -> dict[str, dict[str, str]]:
    ds_metadata_dict = {}
    for ds_attr in ds:
        ds_tag_idx = re.sub("[(,) ]", "", str(ds_attr.tag))
        if ds_tag_idx == "7fe00010":
            continue
        if ds_attr.VR != "SQ":
            value = str(ds_attr.value)
        else:
            value = []  # type: ignore[assignment]
            for inner_ds_idx in range(ds[ds_tag_idx].VM):
                value.append(dcm2dictmetadata(ds=ds[ds_tag_idx][inner_ds_idx]))  # type: ignore[attr-defined]

        ds_metadata_dict[ds_tag_idx] = {
            "vr": ds_attr.VR,
            "name": ds_attr.name,
            "value": value,
        }
    return ds_metadata_dict


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount(path="/static", app=StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request) -> HTMLResponse:
    clean_all()
    return templates.TemplateResponse("index.html", {"request": request})


class MedsamLite(nn.Module):
    def __init__(
        self,  # noqa: ANN101
        image_encoder: TinyViT,
        mask_decoder: MaskDecoder,
        prompt_encoder: PromptEncoder,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(
        self,  # noqa: ANN101
        image: torch.Tensor,
        box_np: torch.Tensor,
    ) -> torch.Tensor:
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device="cpu")
            two_d = 2
            if len(box_torch.shape) == two_d:
                box_torch = box_torch[:, None, :]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks  # type: ignore[no-any-return]

    @torch.no_grad()
    def postprocess_masks(
        self,  # noqa: ANN101
        masks: torch.Tensor,
        new_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        masks = masks[..., : new_size[0], : new_size[1]]
        return functional.interpolate(  # type: ignore[no-any-return]
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )


@lru_cache(maxsize=1)
def load_model() -> MedsamLite:
    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )
    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
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
    medsam_model = MedsamLite(
        image_encoder=medsam_lite_image_encoder,
        mask_decoder=medsam_lite_mask_decoder,
        prompt_encoder=medsam_lite_prompt_encoder,
    )
    medsam_lite_checkpoint = torch.load("./prm/lite_medsam.pth", map_location="cpu")
    medsam_model.load_state_dict(medsam_lite_checkpoint)
    medsam_model.to("cpu")

    return medsam_model


def image_preprocessing(
    img: NDArray[Any],
    downscale_dimensionality: int,
    *,
    multichannel: bool = True,
    retain_aspect_ratio: bool = False,
) -> NDArray[Any]:
    img = (255.0 * ((img - np.min(img)) / (np.max(img) - np.min(img)))).astype(np.uint8)
    if downscale_dimensionality:
        if retain_aspect_ratio:
            aspr = img.shape[1] / img.shape[0]
            h = min([downscale_dimensionality, img.shape[0], img.shape[1]])
            w = int(h * aspr)
            new_shape = (h, w)
        else:
            new_shape = (
                min([downscale_dimensionality, img.shape[0]]),
                min([downscale_dimensionality, img.shape[1]]),
            )
        img = cv2.resize(img, (new_shape[1], new_shape[0]))
    two_d = 2
    if (multichannel) and (len(img.shape) == two_d):
        img = np.stack(3 * [img], axis=-1)
    return img


@lru_cache(maxsize=2**32)
def cache_bbox_img(dcm_hash: str) -> str:
    fp = Path("./tmp/session-data/clean") / (dcm_hash + "_bbox.png")
    if not fp.exists():
        return None
    bbox_pil_img = Image.open(fp)
    bbox_img_buf = BytesIO()
    bbox_pil_img.save(bbox_img_buf, format="PNG")
    return base64.b64encode(bbox_img_buf.getvalue()).decode("utf-8")


@app.post("/conversion_info")
async def conversion_info(dicom_pair_fp: list[str]) -> dict:
    dcm_hash = dicom_pair_fp[1].split("/")[-1].split(".")[0]
    downscale_dimensionality = 1024
    raw_dcm = pydicom.dcmread(dicom_pair_fp[0])
    raw_img = image_preprocessing(
        raw_dcm.pixel_array,
        downscale_dimensionality=downscale_dimensionality,
        multichannel=True,
        retain_aspect_ratio=True,
    )
    raw_buf = BytesIO()
    Image.fromarray(raw_img).save(raw_buf, format="PNG")
    raw_img_base64 = base64.b64encode(raw_buf.getvalue()).decode("utf-8")
    cleaned_dcm = pydicom.dcmread(dicom_pair_fp[1])
    cleaned_img = image_preprocessing(
        cleaned_dcm.pixel_array,
        downscale_dimensionality=downscale_dimensionality,
        multichannel=True,
        retain_aspect_ratio=True,
    )
    cleaned_buf = BytesIO()
    Image.fromarray(cleaned_img).save(cleaned_buf, format="PNG")
    cleaned_img_base64 = base64.b64encode(cleaned_buf.getvalue()).decode("utf-8")
    if cache_bbox_img(dcm_hash=dcm_hash) is None:
        bboxes_dicom_img = raw_img_base64
    else:
        bboxes_dicom_img = cache_bbox_img(dcm_hash=dcm_hash)
    return {
        "raw_dicom_metadata": dcm2dictmetadata(ds=raw_dcm),
        "raw_dicom_img_data": raw_img_base64,
        "cleaned_dicom_metadata": dcm2dictmetadata(ds=cleaned_dcm),
        "cleaned_dicom_img_data": cleaned_img_base64,
        "bboxes_dicom_img_data": bboxes_dicom_img,
    }


@app.post("/get_mask_from_file/")
async def get_mask_from_file(current_dcm_fp: str = Body(...)) -> MaskFromFileResponse:
    current_dcm = pydicom.dcmread(current_dcm_fp)
    return MaskFromFileResponse(
        PixelData=base64.b64encode(current_dcm.SegmentSequence[0].PixelData).decode(
            "utf-8",
        ),
        dimensions=[current_dcm.Columns, current_dcm.Rows],
    )


@app.post("/modify_dicom/")
async def modify_dicom(data: DicomData) -> ModifyResponse:
    pixel_data = base64.b64decode(data.pixel_data)
    filepath = data.filepath
    modified_dcm = pydicom.dcmread(filepath)
    modified_dcm.SegmentSequence[0].PixelData = pixel_data
    modified_dcm.SegmentSequence[0].SegmentDescription = ";".join(data.classes)
    modified_dcm.save_as(filepath)
    return ModifyResponse(success=True)


@app.post("/upload_files/", name="upload_files")
async def get_files(files: list[UploadFile]) -> UploadFilesResponse:
    clean_imgs()
    proper_dicom_paths = []
    total_uploaded_file_bytes = 0
    for file in files:
        contents = await file.read()
        fp = Path("./tmp/session-data/raw/" + file.filename.split("/")[-1])  # type: ignore[union-attr]
        with fp.open("wb") as f:
            f.write(contents)
        try:
            pydicom.dcmread(fp)
            proper_dicom_paths.append(fp)
            total_uploaded_file_bytes += len(contents)
        except InvalidDicomError:
            inv_fp = Path(fp)
            inv_fp.unlink()
    total_uploaded_file_megabytes = "%.1f" % (total_uploaded_file_bytes / (10**3) ** 2)
    return UploadFilesResponse(
        n_uploaded_files=len(proper_dicom_paths),
        total_size=total_uploaded_file_megabytes,
    )


def attach_segm_data(
    dcm: pydicom.dataset.FileDataset,
    seg_mask: np.array,  # type: ignore[valid-type]
    class_names: list[str],
) -> pydicom.dataset.FileDataset:
    if type(seg_mask[0, 0]) != np.uint8:  # type: ignore[index]
        msg = "E: Incompatible element-wise data type"
        raise TypeError(msg)
    seg_dataset = pydicom.dataset.Dataset()
    img = dcm.pixel_array
    two_d = 2
    if len(img.shape) != two_d:
        msg = "E: Incompatible image shape"
        raise ValueError(msg)
    seg_dataset.Rows, seg_dataset.Columns = img.shape
    seg_dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
    seg_dataset.BitsAllocated = 8
    seg_dataset.SegmentDescription = ";".join(class_names)
    seg_dataset.PixelData = seg_mask.tobytes()  # type: ignore[attr-defined]
    dcm.SegmentSequence = pydicom.sequence.Sequence([seg_dataset])
    return dcm


def renew_segm_seq(fps: list[str], classes: list[str]) -> None:
    if classes != ["background"]:
        pass
    else:
        pass
    for fp in fps:
        dcm = pydicom.dcmread(fp)
        img_shape = dcm.pixel_array.shape
        mask = np.zeros(shape=img_shape, dtype=np.uint8)
        dcm = attach_segm_data(dcm=dcm, seg_mask=mask, class_names=classes)
        dcm.save_as(fp)


@app.post("/correct_seg_homogeneity")
async def correct_seg_homogeneity() -> None:
    def segment_sequence_homogeneity_check(fps: list[str]) -> bool:
        for fp in fps:
            try:
                dcm = pydicom.dcmread(fp)
            except InvalidDicomError:
                continue
            try:
                mask = np.frombuffer(
                    dcm.SegmentSequence[0].PixelData,
                    dtype=np.uint8,
                ).reshape((dcm.Rows, dcm.Columns))
                found_classes = dcm.SegmentSequence[0].SegmentDescription.split(";")
                if len(found_classes) != (len(np.unique(mask))):
                    return False
            except Exception:
                logging.exception("Exception occurred")
                return False
        if (
            len(set(found_classes)) > 1
            or dcm.SegmentSequence[0].SegmentDescription.split(";")[0] != "background"
        ):
            return False
        return True

    user_fp = Path("./tmp/session-data/user-options.json")
    with user_fp.open() as file:
        user_input = json.load(file)
    output_fp = Path(user_input["output_dcm_dp"])
    fps = list(output_fp.rglob("*.dcm"))
    homogeneity_state = segment_sequence_homogeneity_check(fps)  # type: ignore[arg-type]
    if not homogeneity_state:
        renew_segm_seq(fps, ["background"])  # type: ignore[arg-type]


@app.post("/get_batch_classes")
async def get_batch_classes() -> dict[str, list[str]]:
    user_fp = Path("./tmp/session-data/user-options.json")
    with user_fp.open() as file:
        user_input = json.load(file)
    output_fp = Path(user_input["output_dcm_dp"])
    fps = list(output_fp.rglob("*.dcm"))
    try:
        found_classes = {
            "classes": pydicom.dcmread(fps[0])
            .SegmentSequence[0]
            .SegmentDescription.split(";"),
        }
    except AttributeError:
        sys.exit("E: Fatal Error; corrupted segmentation sequence attribute detected")
    return found_classes


@app.post("/align_classes")
async def align_classes(classes: list[str]) -> None:
    user_fp = Path("./tmp/session-data/user-options.json")
    with user_fp.open() as file:
        user_input = json.load(file)
    output_fp = Path(user_input["output_dcm_dp"])
    fps = list(output_fp.rglob("*.dcm"))
    renew_segm_seq(fps, classes)  # type: ignore[arg-type]


@app.post("/session", name="session")
async def handle_session_button_click(session_dict: dict[str, Any]) -> None:
    session_fp = Path("./tmp/session-data/session.json")
    with session_fp.open("w") as file:
        json.dump(session_dict, file)


@app.post("/custom_config/", name="custom_config")
async def custom_config(config_file: UploadFile) -> None:
    contents = await config_file.read()
    custom_fp = Path("./tmp/session-data/custom-config.csv")
    with custom_fp.open("wb") as file:
        file.write(contents)


@torch.no_grad()
def medsam_inference(
    medsam_model: MedsamLite,
    img_embed: torch.Tensor,
    box_256: NDArray[Any],
    new_size: tuple[int, int],
    original_size: tuple[int, int],
) -> NDArray[Any]:
    box_torch = torch.as_tensor(box_256, dtype=torch.float, device=img_embed.device)
    two_d = 2
    if len(box_torch.shape) == two_d:
        box_torch = box_torch[:, None, :]
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = medsam_model.postprocess_masks(
        low_res_logits,
        new_size,
        original_size,
    )
    low_res_pred = torch.sigmoid(low_res_pred)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    pred_threshold = 0.5
    return (low_res_pred > pred_threshold).astype(np.uint8)  # type: ignore[no-any-return, attr-defined]


@app.post("/medsam_estimation/")
async def medsam_estimation(boxdata: BoxData) -> BoxDataResponse:
    start = boxdata.normalized_start
    end = boxdata.normalized_end
    seg_class = boxdata.seg_class
    inp_idx = boxdata.inp_idx
    bbox = np.array(
        [
            min(start["x"], end["x"]),
            min(start["y"], end["y"]),
            max(end["x"], start["x"]),
            max(end["y"], start["y"]),
        ],
    )
    box_256 = bbox[None, :] * 256
    time.time()
    medsam_model = load_model()
    temp_dir = Path("./tmp/session-data/embed")
    embedding = torch.load(temp_dir / f"embed_{inp_idx}.pt")
    hs = np.load(temp_dir / "Hs.npy")
    ws = np.load(temp_dir / "Ws.npy")
    medsam_seg = medsam_inference(
        medsam_model,
        embedding,
        box_256,
        (256, 256),
        (hs[inp_idx], ws[inp_idx]),
    )
    medsam_seg = (seg_class * medsam_seg).astype(np.uint8)
    return BoxDataResponse(
        mask=base64.b64encode(medsam_seg).decode("utf-8"),  # type: ignore[arg-type]
        dimensions=[int(ws[inp_idx]), int(hs[inp_idx])],
    )


def prepare_medsam() -> None:
    medsam_model = load_model()
    raw_fp = Path("./tmp/session-data/raw")
    dcm_fps = sorted(raw_fp.glob("*"))
    time.time()
    temp_dir = Path("./tmp/session-data/embed")
    hs, ws = [], []
    for idx, dcm_fp in enumerate(dcm_fps):
        img = pydicom.dcmread(dcm_fp).pixel_array
        two_d = 2
        img_3c = (
            np.repeat(img[:, :, None], 3, axis=-1) if len(img.shape) == two_d else img
        )
        h, w, _ = img_3c.shape
        hs.append(h)
        ws.append(w)
        img_256 = cv2.resize(src=img_3c, dsize=(256, 256)).astype(np.float32)
        newh, neww = img_256.shape[:2]
        img_256 = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(),
            a_min=1e-8,
            a_max=None,
        )
        img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            embedding = medsam_model.image_encoder(img_256_tensor)
            torch.save(embedding, temp_dir / f"embed_{idx}.pt")

    np.save(temp_dir / "Hs.npy", np.array(hs))
    np.save(temp_dir / "Ws.npy", np.array(ws))


def deidentification_attributes(
    user_input: dict[str, Any],
    dcm: pydicom.dataset.FileDataset,
) -> pydicom.dataset.FileDataset:
    user_input_lookup_table = {
        "clean_image": "113101",
        "retain_safe_private": "113111",
        "retain_uids": "113110",
        "retain_device_identity": "113109",
        "retain_patient_characteristics": "113108",
        "date_processing": {
            "offset": "113107",
            "keep": "113106",
        },
        "retain_descriptors": "113105",
    }
    deidentification_code_sequence = "DCM:11310"
    if not set(user_input_lookup_table.keys()).issubset(set(user_input.keys())):
        msg = (
            "E: Inconsistency with user input keys"
            "with lookup de-identification table keys"
        )
        raise KeyError(
            msg,
        )
    for option_name in user_input_lookup_table:
        choice = user_input[option_name]
        if option_name == "date_processing":
            if choice in user_input_lookup_table["date_processing"]:
                deidentification_code_sequence += (
                    "/" + user_input_lookup_table["date_processing"][choice]  # type: ignore[index]
                )
        elif choice:
            deidentification_code_sequence += "/" + user_input_lookup_table[option_name]  # type: ignore[operator]
    dcm.add_new(tag=(0x0012, 0x0062), VR="LO", value="YES")
    dcm.add_new(tag=(0x0012, 0x0063), VR="LO", value=deidentification_code_sequence)
    if user_input["clean_image"]:
        dcm.add_new(tag=(0x0028, 0x0301), VR="LO", value="NO")
    return dcm


def bbox_area_remover(
    img: NDArray[Any],
    bboxes: NDArray[Any],
    initial_array_shape: tuple[int],
    downscaled_array_shape: tuple[int],
) -> NDArray[Any]:
    reducted_region_color = np.mean(img).astype(np.uint16)
    multiplicative_mask = np.ones(downscaled_array_shape, dtype=np.uint8)
    additive_mask = np.zeros(initial_array_shape, dtype=np.uint8)
    bbox_mask = np.zeros(downscaled_array_shape, dtype=np.uint8)
    for bbox in bboxes:
        x0, y0 = bbox[0, 0 : (1 + 1)]
        x1, y1 = bbox[1, 0 : (1 + 1)]
        x2, y2 = bbox[2, 0 : (1 + 1)]
        x3, y3 = bbox[3, 0 : (1 + 1)]
        rectangle = np.array([[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]], dtype=np.int32)
        cv2.polylines(bbox_mask, rectangle, isClosed=True, color=1, thickness=2)
        cv2.fillPoly(multiplicative_mask, rectangle, 0)
    multiplicative_mask = cv2.resize(
        multiplicative_mask,
        (initial_array_shape[1], initial_array_shape[0]),  # type: ignore[misc]
        interpolation=cv2.INTER_NEAREST,
    )
    bbox_mask = cv2.resize(
        bbox_mask,
        (initial_array_shape[1], initial_array_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    additive_mask = reducted_region_color * (multiplicative_mask == 0)
    cleaned_img = img * multiplicative_mask + additive_mask
    bbox_img = np.maximum(bbox_mask * np.max(img), img)
    return cleaned_img, bbox_img


def image_deintentifier(
    dcm: pydicom.dataset.FileDataset,
) -> pydicom.dataset.FileDataset:
    min_dim = 50
    downscale_dimensionality = 1024
    if downscale_dimensionality < min_dim:
        msg = "E: Downscale dimensionality is excessively small"
        raise ValueError(msg)
    dcm.decompress()
    raw_img_uint16_grayscale = dcm.pixel_array
    if min(raw_img_uint16_grayscale.shape) < min_dim:
        return dcm
    if downscale_dimensionality < max(
        raw_img_uint16_grayscale.shape[0],
        raw_img_uint16_grayscale.shape[1],
    ):
        pass

    raw_img_uint8_rgb = image_preprocessing(
        img=raw_img_uint16_grayscale,
        downscale_dimensionality=downscale_dimensionality,
    )

    pipeline = keras_ocr.detection.Detector()
    bboxes = pipeline.detect([raw_img_uint8_rgb])[0]
    initial_array_shape = raw_img_uint16_grayscale.shape
    downscaled_array_shape = raw_img_uint8_rgb.shape[:-1]
    bbox_img = raw_img_uint16_grayscale.copy()
    if np.size(bboxes) != 0:
        cleaned_img, bbox_img = bbox_area_remover(
            img=raw_img_uint16_grayscale,
            bboxes=bboxes,
            initial_array_shape=initial_array_shape,  # type: ignore[arg-type]
            downscaled_array_shape=downscaled_array_shape,  # type: ignore[arg-type]
        )
        dcm.PixelData = cleaned_img.tobytes()
    else:
        pass
    return dcm, bbox_img


def merge_action(
    primary_srs: pd.core.series.Series,
    action2beassigned_srs: pd.core.series.Series,
) -> pd.core.series.Series:
    return primary_srs.where(
        cond=action2beassigned_srs.isna(),
        other=action2beassigned_srs,
        axis=0,
        inplace=False,
    )


def merge_with_custom_user_config_file(
    requested_action_group_df: pd.core.frame.DataFrame,
    custom_config_df: pd.core.frame.DataFrame,
) -> pd.core.frame.DataFrame:
    valid_actions = {"X", "K", "C"}
    if not set(custom_config_df["Action"]).issubset(valid_actions):
        sys.exit()
    requested_action_group_df = requested_action_group_df.merge(
        custom_config_df[["Action"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    requested_action_group_df.loc[
        requested_action_group_df["Action"].isin(["X", "K", "C"]),
        "Requested Action Group",
    ] = requested_action_group_df["Action"]
    return requested_action_group_df.drop(columns=["Action"])


def get_action_group(
    user_input: dict[str, Any],
    action_groups_df: pd.core.frame.DataFrame,
    custom_config_df: pd.core.frame.DataFrame | None,
) -> pd.core.frame.DataFrame:
    requested_action_group_df = pd.DataFrame(
        data=action_groups_df["Default"].to_list(),
        columns=["Requested Action Group"],
        index=action_groups_df.index,
    )
    requested_action_group_df.insert(
        loc=0,
        column="Name",
        value=action_groups_df["Name"].to_list(),
    )
    if user_input["retain_safe_private"]:
        requested_action_group_df["Requested Action Group"] = merge_action(
            primary_srs=requested_action_group_df["Requested Action Group"],
            action2beassigned_srs=action_groups_df["Rtn. Safe Priv. Opt."],
        )
    if user_input["retain_uids"]:
        requested_action_group_df["Requested Action Group"] = merge_action(
            primary_srs=requested_action_group_df["Requested Action Group"],
            action2beassigned_srs=action_groups_df["Rtn. UIDs Opt."],
        )
    if user_input["retain_device_identity"]:
        requested_action_group_df["Requested Action Group"] = merge_action(
            primary_srs=requested_action_group_df["Requested Action Group"],
            action2beassigned_srs=action_groups_df["Rtn. Dev. Id. Opt."],
        )
    if user_input["retain_patient_characteristics"]:
        requested_action_group_df["Requested Action Group"] = merge_action(
            primary_srs=requested_action_group_df["Requested Action Group"],
            action2beassigned_srs=action_groups_df["Rtn. Pat. Chars. Opt."],
        )
    if user_input["date_processing"] == "keep":
        requested_action_group_df["Requested Action Group"] = merge_action(
            primary_srs=requested_action_group_df["Requested Action Group"],
            action2beassigned_srs=action_groups_df["Rtn. Long. Modif. Dates Opt."],
        )
    elif user_input["date_processing"] == "offset":
        requested_action_group_df["Requested Action Group"] = merge_action(
            primary_srs=requested_action_group_df["Requested Action Group"],
            action2beassigned_srs=action_groups_df["Offset Long. Modif. Dates Opt."],
        )
    elif user_input["date_processing"] == "remove":
        requested_action_group_df["Requested Action Group"] = merge_action(
            primary_srs=requested_action_group_df["Requested Action Group"],
            action2beassigned_srs=action_groups_df["Remove Long. Modif. Dates Opt."],
        )
    if user_input["retain_descriptors"]:
        requested_action_group_df["Requested Action Group"] = merge_action(
            primary_srs=requested_action_group_df["Requested Action Group"],
            action2beassigned_srs=action_groups_df["Rtn. Desc. Opt."],
        )
    if type(custom_config_df) == pd.core.frame.DataFrame:
        requested_action_group_df = merge_with_custom_user_config_file(
            requested_action_group_df,
            custom_config_df,
        )
    return requested_action_group_df


def adjust_dicom_metadata(  # noqa: C901
    dcm: pydicom.dataset.FileDataset,
    action_group_fp: str,
    patient_pseudo_id: str,  # noqa: ARG001
    days_total_offset: int,
    seconds_total_offset: int,
) -> tuple[pydicom.dataset.FileDataset, dict[str, int]]:
    def add_date_offset(input_date_str: str, days_total_offset: str) -> str:
        input_date = datetime.datetime.strptime(input_date_str, "%Y%m%d").astimezone(
            datetime.timezone.utc,
        )
        output_date = input_date + datetime.timedelta(days=days_total_offset)  # type: ignore[arg-type]
        return output_date.strftime("%Y%m%d")  # type: ignore[no-any-return]

    def seconds2daytime(seconds_total_offset: int) -> str:
        output_hours = seconds_total_offset // 3600
        output_minutes = (seconds_total_offset % 3600) // 60
        output_seconds = (seconds_total_offset % 3600) % 60
        return "%.2d%.2d%.2d" % (
            output_hours,
            output_minutes,
            output_seconds,
        )

    def recursive_sq_cleaner(  # noqa: C901
        ds: pydicom.dataset.FileDataset,
        action: str,
        action_attr_tag_idx: str,
    ) -> pydicom.dataset.FileDataset:
        for ds_attr in ds:
            ds_tag_idx = re.sub("[(,) ]", "", str(ds_attr.tag))
            if ds[ds_tag_idx].VR == "SQ":
                for inner_ds_idx in range(ds[ds_tag_idx].VM):
                    ds[ds_tag_idx].value[inner_ds_idx] = recursive_sq_cleaner(
                        ds=ds[ds_tag_idx][inner_ds_idx],
                        action=action,
                        action_attr_tag_idx=action_attr_tag_idx,
                    )
            elif action_attr_tag_idx == ds_tag_idx:
                if action == "Z":
                    # Check if ds_tag_idx is not one of the specified tags
                    if ds_tag_idx not in ["00100010", "00100020"]:
                        msg = (
                            "E: Cannot apply action code `Z` in any"
                            "other attribute besides Patient ID and"
                            "Patient Name; the issue is likely on "
                            "the action group config object"
                        )
                        raise ValueError(
                            msg,
                        )
                elif action == "X":
                    ds[ds_tag_idx].value = ""
                elif action == "C":
                    if ds[ds_tag_idx].value != "" and ds[ds_tag_idx].VR == "DA":
                        tag_value_replacements["days_total_offset"] = days_total_offset
                        ds[ds_tag_idx].value = add_date_offset(
                            input_date_str=ds[ds_tag_idx].value,
                            days_total_offset=days_total_offset,  # type: ignore[arg-type]
                        )
                    elif ds[ds_tag_idx].VR == "TM":
                        tag_value_replacements[
                            "seconds_total_offset"
                        ] = seconds_total_offset
                        ds[ds_tag_idx].value = seconds2daytime(
                            seconds_total_offset=tag_value_replacements[
                                "seconds_total_offset"
                            ],
                        )
        return ds

    action_group_df = pd.read_csv(filepath_or_buffer=action_group_fp, index_col=0)
    tag_value_replacements = {}
    tag_value_replacements["days_total_offset"] = 0
    tag_value_replacements["seconds_total_offset"] = 0
    for action_attr_tag_idx in action_group_df.index:
        action = action_group_df.loc[action_attr_tag_idx].iloc[1]
        dcm = recursive_sq_cleaner(
            ds=dcm,
            action=action,
            action_attr_tag_idx=action_attr_tag_idx,
        )
    return dcm, tag_value_replacements


class Rwdcm:
    def __init__(
        self,  # noqa: ANN101
        in_dp: str,
        out_dp: str,
    ) -> None:
        self.SAFETY_SWITCH = True
        if not self.SAFETY_SWITCH:
            pass
        if in_dp[-1] != "/":
            in_dp = in_dp + "/"
        self.raw_data_dp = in_dp
        self.raw_dicom_paths = sorted(self.get_dicom_paths(data_dp=self.raw_data_dp))
        self.dicom_pair_fps = []
        self.out_dp = out_dp
        self.clean_data_dp = out_dp + "/" + "de-identified-files/"
        already_cleaned_dicom_paths = str(
            self.get_dicom_paths(data_dp=self.clean_data_dp),
        )
        self.hashes_of_already_converted_files = [
            already_cleaned_dicom_path.split("/")[-1].split(".")[0]
            for already_cleaned_dicom_path in already_cleaned_dicom_paths
        ]
        self.n_dicom_files = len(self.raw_dicom_paths)
        self.DICOM_IDX = -1

    def __next__(
        self,  # noqa: ANN101
    ) -> bool:
        self.DICOM_IDX += 1
        if self.n_dicom_files - 1 >= self.DICOM_IDX:
            self.raw_dicom_path = self.raw_dicom_paths[self.DICOM_IDX]
            return True
        return False

    def get_dicom_paths(
        self,  # noqa: ANN101
        data_dp: str,
    ) -> list[Path]:
        dicom_paths = list(Path(data_dp).rglob("*"))
        proper_dicom_paths = []
        for dicom_path in dicom_paths:
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            if ds:
                proper_dicom_paths.append(dicom_path)
        return proper_dicom_paths

    def parse_file(
        self,  # noqa: ANN101
    ) -> pydicom.dataset.FileDataset | bool:
        self.input_dicom_hash = hashlib.sha256(
            str(self.raw_dicom_path).encode("UTF-8"),
        ).hexdigest()
        if self.input_dicom_hash in self.hashes_of_already_converted_files:
            return False
        return pydicom.dcmread(self.raw_dicom_path)

    def export_processed_file(
        self,  # noqa: ANN101
        dcm: pydicom.dataset.FileDataset,
        bbox_img: NDArray[Any],
    ) -> None:
        self.clean_dicom_dp = (
            self.clean_data_dp
            + str(dcm[0x0010, 0x0020].value)
            + "/"
            + str(dcm[0x0008, 0x0060].value)
            + "/"
            + str(dcm[0x0020, 0x0011].value)
        )
        if not Path(self.clean_dicom_dp).exists():
            Path(self.clean_dicom_dp).mkdir(parents=True)
        if bbox_img is not None:
            bbox_img_fp = self.out_dp + "/" + self.input_dicom_hash + "_bbox" + ".png"
            Image.fromarray(bbox_img).save(bbox_img_fp)
            cache_bbox_img(self.input_dicom_hash)
        clean_dicom_fp = self.clean_dicom_dp + "/" + self.input_dicom_hash + ".dcm"
        dcm.save_as(clean_dicom_fp)
        self.dicom_pair_fps.append((self.raw_dicom_path, clean_dicom_fp))

    def export_session(
        self,  # noqa: ANN101
        session: dict[str, dict[str, str]],
    ) -> None:
        session_fp = Path(self.clean_data_dp + "/session.json")
        with session_fp.open("w") as file:
            json.dump(session, file)


def dicom_deidentifier(  # noqa: PLR0912, PLR0915
    session_filepath: None | str = None,
) -> tuple[dict[str, dict[str, str]], list[tuple[str]]]:
    if Path("./tmp/session-data/custom-config.csv").is_file():
        custom_config_df = pd.read_csv(
            filepath_or_buffer="./tmp/session-data/custom-config.csv",
            index_col=0,
        )
        custom_config_df.index = custom_config_df.index.str.strip("'")
    else:
        custom_config_df = None
    action_groups_df = pd.read_csv(
        filepath_or_buffer="./python/tmp/action-groups-dcm.csv",
        index_col=0,
    )
    if session_filepath is None or not Path(session_filepath).is_file():
        session = {}
    else:
        with Path("./tmp/session-data/session.json").open() as file:
            session = json.load(file)
    if Path("./tmp/session-data/user-options.json").is_file():
        with Path("./tmp/session-data/user-options.json").open() as file:
            user_input = json.load(file)
    else:
        sys.exit("E: No client de-identification configuration was provided")
    pseudo_patient_ids = []
    pseudo_patient_ids = [
        int(patient_deidentification_properties["patient_pseudo_id"])
        for patient_deidentification_properties in session.values()
    ]
    max_pseudo_patient_id = -1 if pseudo_patient_ids == [] else max(pseudo_patient_ids)
    requested_action_group_df = get_action_group(
        user_input=user_input,
        action_groups_df=action_groups_df,
        custom_config_df=custom_config_df,
    )
    requested_action_group_df.to_csv(
        "./tmp/session-data/requested-action-group-dcm.csv",
    )
    rw_obj = Rwdcm(in_dp=user_input["input_dcm_dp"], out_dp=user_input["output_dcm_dp"])
    while next(rw_obj):
        dcm = rw_obj.parse_file()
        if dcm is False:
            msg = "E: DICOM file is corrupted or missing"
            raise ValueError(msg)
        date_processing_choices = {"keep", "offset", "remove"}
        if user_input["date_processing"] not in date_processing_choices:
            msg = "E: Invalid date processing input"
            raise ValueError(msg)
        real_patient_id = dcm[0x0010, 0x0020].value  # type: ignore[index]
        patient_deidentification_properties = session.get(real_patient_id, False)
        if not patient_deidentification_properties:
            max_pseudo_patient_id += 1
            session[real_patient_id] = {
                "patient_pseudo_id": "%.6d" % max_pseudo_patient_id,
            }
            days_total_offset = secrets.randbelow((2 * 10 * 365) - (10 * 365) + 1) + (
                10 * 365
            )
            seconds_total_offset = secrets.randbelow(24 * 60 * 60)
        else:
            days_total_offset = session[real_patient_id]["days_offset"]
            seconds_total_offset = session[real_patient_id]["seconds_offset"]
        dcm, tag_value_replacements = adjust_dicom_metadata(
            dcm=dcm,  # type: ignore[arg-type]
            action_group_fp="./tmp/session-data/requested-action-group-dcm.csv",
            patient_pseudo_id=session[real_patient_id]["patient_pseudo_id"],
            days_total_offset=days_total_offset,
            seconds_total_offset=seconds_total_offset,
        )
        session[real_patient_id]["days_offset"] = tag_value_replacements[
            "days_total_offset"
        ]
        session[real_patient_id]["seconds_offset"] = tag_value_replacements[
            "seconds_total_offset"
        ]
        dcm = deidentification_attributes(user_input=user_input, dcm=dcm)
        if user_input["clean_image"]:
            dcm, bbox_img = image_deintentifier(dcm=dcm)
            bbox_img = image_preprocessing(
                bbox_img,
                downscale_dimensionality=max(bbox_img.shape),
                multichannel=True,
                retain_aspect_ratio=True,
            )
        else:
            bbox_img = None
        rw_obj.export_processed_file(dcm=dcm, bbox_img=bbox_img)
        rw_obj.export_session(session=session)
    return session, rw_obj.dicom_pair_fps


@app.post("/submit_button")
async def handle_submit_button_click(user_options: UserOptionsClass) -> list[Any]:
    user_options = dict(user_options)  # type: ignore[assignment]
    dp, _, fps = next(iter(os.walk("./tmp/session-data/raw")))
    if set(fps).issubset({".gitkeep"}):
        return False  # type: ignore[return-value]
    default_options = {
        "input_dcm_dp": "./tmp/session-data/raw",
        "output_dcm_dp": "./tmp/session-data/clean",
        "clean_image": True,
        "retain_safe_private": False,
        "retain_uids": False,
        "retain_device_identity": False,
        "retain_patient_characteristics": False,
        "date_processing": "offset",
        "retain_descriptors": False,
        "patient_pseudo_id_prefix": "<PREFIX ID> - ",
    }
    user_options["input_dcm_dp"] = default_options["input_dcm_dp"]  # type: ignore[index]
    user_options["output_dcm_dp"] = default_options["output_dcm_dp"]  # type: ignore[index]
    user_fp = Path("./tmp/session-data/user-options.json")
    with user_fp.open("w") as file:
        json.dump(user_options, file)
    session, dicom_pair_fps = dicom_deidentifier(
        session_filepath="./tmp/session-data/session.json",
    )
    session_fp = Path("./tmp/session-data/session.json")
    with session_fp.open("w") as file:
        json.dump(session, file)
    prepare_medsam()
    return dicom_pair_fps


def ndarray_size(arr: NDArray[Any]) -> int:
    return arr.itemsize * arr.size


if __name__ == "__main__":
    if os.getenv("STAGING"):
        tmp_directories = [
            Path("tmp/session-data/raw"),
            Path("tmp/session-data/clean"),
            Path("tmp/session-data/embed"),
        ]
        for directory in tmp_directories:
            directory.mkdir(parents=True, exist_ok=True)
        run(app, host="0.0.0.0", port=8000)  # noqa: S104
