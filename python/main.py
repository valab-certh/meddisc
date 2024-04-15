from __future__ import annotations

import base64
import datetime
import glob
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib import request

import aiofiles
import cv2
import keras_ocr
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import pytest
import requests
import torch
from fastapi import Body, FastAPI, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel
from pydicom.errors import InvalidDicomError
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from torch import nn
from torch.nn import functional
from uvicorn import run

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tmp.tiny_vit_sam import TinyViT


class UserOptionsClass(BaseModel):
    skip_deidentification: bool
    clean_image: bool
    annotation: bool
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


class SegData(BaseModel):
    pixel_data: str
    filepath: str
    classes: list[str]
    n_dicom: int
    dcm_idx: int


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
    skip_deidentification: bool


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
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=templates_dir)
app.mount(path="/static", app=StaticFiles(directory=static_dir), name="static")

client = TestClient(app)


def app_url() -> str:
    return "http://0.0.0.0:443"


def test_upload_files() -> None:
    with Path("./prm/1-1.dcm").open("rb") as file:
        files = {"files": ("./prm/1-1.dcm", file, "application/dicom")}
        response = client.post(app_url() + "/upload_files", files=files)
        if response.status_code != status.HTTP_200_OK:
            raise AssertionError
        UploadFilesResponse.model_validate(response.json())


def test_submit_button() -> None:
    test_options = {
        "skip_deidentification": False,
        "clean_image": True,
        "annotation": False,
        "retain_safe_private": False,
        "retain_uids": False,
        "retain_device_identity": False,
        "retain_patient_characteristics": False,
        "date_processing": "remove",
        "retain_descriptors": False,
        "patient_pseudo_id_prefix": "OrgX - ",
    }
        response = client.post(self.app_url() + "/submit_button", json=test_options)
        if response.status_code != status.HTTP_200_OK:
            raise AssertionError
        json_response = response.json()
    desired_hash = "cd6e8eae4006ca7b150c3217667de6b6f7b435f93961d182e72b4da7773884a9"
    hasher = hashlib.sha256()
    block_size = 65536
    with Path(json_response[0][1]).open("rb") as file:
        buf = file.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = file.read(block_size)
    generated_hash = hasher.hexdigest()
    if desired_hash != generated_hash:
        msg = "E: Generated hash doesn't match"
        raise ValueError(
            msg,
        )


@app.get("/check_existence_of_clean")
async def check_existence_of_clean() -> UploadFilesResponse:
    session_fp = "./tmp/session-data/clean/de-identified-files/session.json"
    proper_dicom_paths = sorted(
        glob.glob(  # noqa: PTH207
            "./tmp/session-data/clean/de-identified-files/**/*.dcm",
            recursive=True,
        ),
    )
    total_uploaded_file_bytes = 0
    skip_deidentification = True
    if os.path.exists(session_fp) and len(proper_dicom_paths) != 0:  # noqa: PTH110
        for dcm_fp in proper_dicom_paths:
            with open(file=dcm_fp, mode="br") as f:  # noqa: ASYNC101, PTH123
                total_uploaded_file_bytes += len(f.read())
            total_uploaded_file_megabytes = "%.1f" % (
                total_uploaded_file_bytes / ((10**3) ** 2)
            )
    else:
        total_uploaded_file_megabytes = "0.0"

    return UploadFilesResponse(
        n_uploaded_files=len(proper_dicom_paths),
        total_size=total_uploaded_file_megabytes,
        skip_deidentification=skip_deidentification,
    )


@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request) -> HTMLResponse:
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
    from tmp.tiny_vit_sam import TinyViT

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
    medsam_dir = Path(__file__).parent / "templates" / "lite_medsam.pth"
    medsam_lite_checkpoint = torch.load(medsam_dir, map_location="cpu")
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
        return None  # type: ignore[return-value]
    bbox_pil_img = Image.open(fp)
    bbox_img_buf = BytesIO()
    bbox_pil_img.save(bbox_img_buf, format="PNG")
    return base64.b64encode(bbox_img_buf.getvalue()).decode("utf-8")


@app.post("/conversion_info")
async def conversion_info(dicom_pair_fp: list[str]) -> dict:  # type: ignore[type-arg]
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
    Image.fromarray(raw_img).save(raw_buf, format="PNG")  # type: ignore[no-untyped-call]
    cleaned_dcm = pydicom.dcmread(dicom_pair_fp[1])
    cleaned_img = image_preprocessing(
        cleaned_dcm.pixel_array,
        downscale_dimensionality=downscale_dimensionality,
        multichannel=True,
        retain_aspect_ratio=True,
    )
    cleaned_buf = BytesIO()
    Image.fromarray(cleaned_img).save(cleaned_buf, format="PNG")  # type: ignore[no-untyped-call]
    cleaned_img_base64 = base64.b64encode(cleaned_buf.getvalue()).decode("utf-8")
    if cache_bbox_img(dcm_hash=dcm_hash) is None:
        pass
    else:
        cache_bbox_img(dcm_hash=dcm_hash)
    return {
        "raw_dicom_metadata": dcm2dictmetadata(ds=raw_dcm),
        "cleaned_dicom_metadata": dcm2dictmetadata(ds=cleaned_dcm),
        "cleaned_dicom_img_data": cleaned_img_base64,
    }


def get_raw_dicom_img_data_(fp):  # type: ignore[no-untyped-def] # noqa: ANN201, ANN001
    downscale_dimensionality = 1024
    raw_dcm = pydicom.dcmread(fp)
    raw_img = image_preprocessing(
        raw_dcm.pixel_array,
        downscale_dimensionality=downscale_dimensionality,
        multichannel=True,
        retain_aspect_ratio=True,
    )
    raw_buf = BytesIO()
    Image.fromarray(raw_img).save(raw_buf, format="PNG")  # type: ignore[no-untyped-call]
    return base64.b64encode(raw_buf.getvalue()).decode("utf-8")


@app.post("/get_raw_dicom_img_data/")
async def get_raw_dicom_img_data(dicom_pair_fp: list[str]) -> dict:  # type: ignore[type-arg]
    raw_img_base64 = get_raw_dicom_img_data_(dicom_pair_fp[0])  # type: ignore[no-untyped-call]
    return {"raw_dicom_img_data": raw_img_base64}


@app.post("/get_bboxes_dicom_img_data/")
async def get_bboxes_dicom_img_data(dicom_pair_fp: list[str]) -> dict:  # type: ignore[type-arg]
    raw_img_base64 = get_raw_dicom_img_data_(dicom_pair_fp[0])  # type: ignore[no-untyped-call]
    dcm_hash = dicom_pair_fp[1].split("/")[-1].split(".")[0]
    if cache_bbox_img(dcm_hash=dcm_hash) is None:
        bboxes_dicom_img = raw_img_base64
    else:
        bboxes_dicom_img = cache_bbox_img(dcm_hash=dcm_hash)
    return {"bboxes_dicom_img_data": bboxes_dicom_img}


def generate_nifti_info() -> tuple[list[dict], dict]:  # type: ignore[type-arg]
    user_fp = Path("./tmp/session-data/user-options.json")
    with user_fp.open() as file:
        user_input = json.loads(file.read())
    output_dp = user_input["output_dcm_dp"]
    dcm_fps = list(glob.glob(os.path.join(output_dp, "**/*.dcm"), recursive=True))  # noqa: PTH207, PTH118
    series_info = {}
    for dcm_fp in sorted(dcm_fps):
        dcm = pydicom.dcmread(dcm_fp)
        series_dp = "/".join(str(dcm_fp).split("/")[:-1])
        if series_dp not in series_info:
            series_info[series_dp] = {
                "dcm_fps": [],
                "height": dcm.Rows,
                "width": dcm.Columns,
            }
        series_info[series_dp]["dcm_fps"].append(dcm_fp)
    nifti_info = []  # type: ignore[var-annotated]
    dicom_fps_to_segm_info = {}
    for series_dp in sorted(series_info.keys()):
        dcm_fps_ = glob.glob(os.path.join(series_dp, "*.dcm"))  # noqa: PTH207, PTH118
        if series_dp.split("/")[-1] != "None":
            nifti_info.append({})
            nifti_info[-1]["fp"] = os.path.join(series_dp, "segmentation.nii")  # noqa: PTH118
            nifti_info[-1]["n_slices"] = len(
                glob.glob(os.path.join(series_dp, "*.dcm")),  # noqa: PTH207, PTH118
            )
            nifti_info[-1]["series_number"] = series_dp.split("/")[-1]
            dcm = pydicom.dcmread(glob.glob(os.path.join(series_dp, "*.dcm"))[0])  # noqa: PTH207, PTH118
            nifti_info[-1]["height"] = dcm.Rows
            nifti_info[-1]["width"] = dcm.Columns
            nifti_info[-1]["dicom_fps"] = []
            for dcm_series_idx, dcm_fp in enumerate(sorted(dcm_fps_)):
                dicom_fps_to_segm_info[dcm_fp] = {
                    "height": dcm.Rows,
                    "width": dcm.Columns,
                    "slice": dcm_series_idx,
                    "nifti_fp": nifti_info[-1]["fp"],
                }
                nifti_info[-1]["dicom_fps"].append(dcm_fp)
            nifti_info[-1]["dicom_fps"] = sorted(nifti_info[-1]["dicom_fps"])
        else:
            for dcm_fp in sorted(dcm_fps_):
                nifti_info.append({})
                nifti_info[-1]["fp"] = ".".join(dcm_fp.split(".")[:-1]) + ".nii"
                nifti_info[-1]["n_slices"] = 1
                nifti_info[-1]["series_number"] = series_dp.split("/")[-1]
                dcm = pydicom.dcmread(dcm_fp)
                nifti_info[-1]["height"] = dcm.Rows
                nifti_info[-1]["width"] = dcm.Columns
                dicom_fps_to_segm_info[dcm_fp] = {
                    "height": dcm.Rows,
                    "width": dcm.Columns,
                    "slice": 0,
                    "nifti_fp": nifti_info[-1]["fp"],
                }
    return nifti_info, dicom_fps_to_segm_info


@app.post("/get_mask_from_file/")
async def get_mask_from_file(current_dcm_fp: str = Body(...)) -> MaskFromFileResponse:
    _, dicom_fps_to_segm_info = generate_nifti_info()
    nifti_fp = dicom_fps_to_segm_info[current_dcm_fp]["nifti_fp"]
    slice_ = dicom_fps_to_segm_info[current_dcm_fp]["slice"]
    h = dicom_fps_to_segm_info[current_dcm_fp]["height"]
    w = dicom_fps_to_segm_info[current_dcm_fp]["width"]
    mask = nib.load(nifti_fp).get_fdata()[..., slice_].astype(np.uint8).copy(order="C")  # type: ignore[attr-defined]
    return MaskFromFileResponse(
        PixelData=base64.b64encode(mask).decode("utf-8"),
        dimensions=[w, h],
    )


@app.post("/export_masks/")
async def export_masks(data: SegData) -> ModifyResponse:
    _, dicom_fps_to_segm_info = generate_nifti_info()
    segm_info_slice = dicom_fps_to_segm_info[data.filepath]
    mask = np.frombuffer(base64.b64decode(data.pixel_data), dtype=np.uint8).reshape(
        (segm_info_slice["height"], segm_info_slice["width"]),
    )
    nifti = nib.load(segm_info_slice["nifti_fp"])  # type: ignore[attr-defined]
    masks = nifti.get_fdata().astype(np.uint8)  # type: ignore[attr-defined]
    masks[..., segm_info_slice["slice"]] = mask
    nifti = nib.Nifti1Image(masks, np.eye(4))  # type: ignore[no-untyped-call, attr-defined]
    nib.save(nifti, segm_info_slice["nifti_fp"])  # type: ignore[attr-defined]

    return ModifyResponse(success=True)


@app.post("/upload_files/", name="upload_files")
async def get_files(files: list[UploadFile]) -> UploadFilesResponse:  # noqa: C901, PLR0912, PLR0915
    proper_dicom_paths = []
    total_uploaded_file_bytes = 0
    skip_deidentification = False
    for file in files:
        if (
            file.filename.split("/")[0] == "de-identified-files"  # type: ignore[union-attr]
            and file.filename.split("/")[-1] == "session.json"  # type: ignore[union-attr]
        ):
            contents = await file.read()
            fp = Path(
                os.path.join(  # noqa: PTH118
                    "./tmp/session-data/clean/",
                    "/".join(file.filename.split("/")[-2:]),  # type: ignore[union-attr]
                ),
            )
            dp = Path(
                os.path.join(  # noqa: PTH118
                    "./tmp/session-data/clean/",
                    "/".join(file.filename.split("/")[-2:-1]),  # type: ignore[union-attr]
                ),
            )
            if not os.path.exists(dp):  # noqa: PTH110
                os.makedirs(dp)  # noqa: PTH103
            async with aiofiles.open(fp, "wb") as f:
                await f.write(contents)
            skip_deidentification = True
    if not skip_deidentification:
        for file in files:
            contents = await file.read()
            fp = Path("./tmp/session-data/raw/" + file.filename.split("/")[-1])  # type: ignore[union-attr]
            async with aiofiles.open(fp, "wb") as f:
                await f.write(contents)
            try:
                dcm = pydicom.dcmread(fp)
                if len(dcm.pixel_array.shape) == 2:  # noqa: PLR2004
                    proper_dicom_paths.append(fp)
                    total_uploaded_file_bytes += len(contents)
                    total_uploaded_file_megabytes = "%.1f" % (
                        total_uploaded_file_bytes / (10**3) ** 2
                    )
                else:
                    fp.unlink()
            except InvalidDicomError:
                fp.unlink()
    else:
        for file in files:
            contents = await file.read()
            if file.filename.split(".")[-1] == "dcm":  # type: ignore[union-attr]
                fp = Path(
                    os.path.join(  # noqa: PTH118
                        "./tmp/session-data/clean/",
                        "/".join(file.filename.split("/")[-5:]),  # type: ignore[union-attr]
                    ),
                )
                proper_dicom_paths.append(fp)
                total_uploaded_file_bytes += len(contents)
                total_uploaded_file_megabytes = "%.1f" % (
                    total_uploaded_file_bytes / (10**3) ** 2
                )
                dp = Path(
                    os.path.join(  # noqa: PTH118
                        "./tmp/session-data/clean/",
                        "/".join(file.filename.split("/")[-5:-1]),  # type: ignore[union-attr]
                    ),
                )
                if not os.path.exists(dp):  # noqa: PTH110
                    os.makedirs(dp)  # noqa: PTH103
                async with aiofiles.open(fp, "wb") as f:
                    await f.write(contents)
            elif file.filename.split(".")[-1] == "nii":  # type: ignore[union-attr]
                fp = Path(
                    os.path.join(  # noqa: PTH118
                        "./tmp/session-data/clean/",
                        "/".join(file.filename.split("/")[-5:]),  # type: ignore[union-attr]
                    ),
                )
                dp = Path(
                    os.path.join(  # noqa: PTH118
                        "./tmp/session-data/clean/",
                        "/".join(file.filename.split("/")[-5:-1]),  # type: ignore[union-attr]
                    ),
                )
                if not os.path.exists(dp):  # noqa: PTH110
                    os.makedirs(dp)  # noqa: PTH103
                async with aiofiles.open(fp, "wb") as f:
                    await f.write(contents)
            elif file.filename.split(".")[-1] == "csv":  # type: ignore[union-attr]
                fp = Path(
                    os.path.join(  # noqa: PTH118
                        "./tmp/session-data/clean/",
                        "/".join(file.filename.split("/")[-2:]),  # type: ignore[union-attr]
                    ),
                )
                dp = Path(
                    os.path.join(  # noqa: PTH118
                        "./tmp/session-data/clean/",
                        "/".join(file.filename.split("/")[-2:-1]),  # type: ignore[union-attr]
                    ),
                )
                if not os.path.exists(dp):  # noqa: PTH110
                    os.makedirs(dp)  # noqa: PTH103
                async with aiofiles.open(fp, "wb") as f:
                    await f.write(contents)

    return UploadFilesResponse(
        n_uploaded_files=len(proper_dicom_paths),
        total_size=total_uploaded_file_megabytes,
        skip_deidentification=skip_deidentification,
    )


def export_classes_to_session(session_classes: list[str]) -> None:
    session_fp = Path("./tmp/session-data/clean/de-identified-files/session.json")
    with session_fp.open() as file:
        session = json.load(file)
    session["classes"] = []
    for class_name in session_classes:
        session["classes"].append(class_name)
    with session_fp.open("w") as file:
        json.dump(session, file, indent=4)


def renew_segmentation_data(classes: list[str]) -> None:
    export_classes_to_session(session_classes=classes)
    nifti_info, _ = generate_nifti_info()
    for nifti_ in nifti_info:
        arr = np.zeros(
            shape=(nifti_["height"], nifti_["width"], nifti_["n_slices"]),
            dtype=np.uint8,
        )
        nifti = nib.Nifti1Image(arr, np.eye(4))  # type: ignore[no-untyped-call, attr-defined]
        nib.save(nifti, nifti_["fp"])  # type: ignore[attr-defined]


@app.post("/export_classes")
def export_classes(classes: list[str]) -> None:
    export_classes_to_session(session_classes=classes)


def get_classes_from_session() -> list[str]:
    session_fp = Path("./tmp/session-data/clean/de-identified-files/session.json")
    with session_fp.open() as file:
        return json.load(file)["classes"]  # type: ignore[no-any-return]


@app.post("/correct_seg_homogeneity")
async def correct_seg_homogeneity() -> None:
    def check_class_names_integrity() -> bool:
        session_fp = Path("./tmp/session-data/clean/de-identified-files/session.json")
        with session_fp.open() as file:
            session = json.load(file)
        if "classes" not in session:
            return False
        if session["classes"][0] != "background":
            return False
        return True

    def check_nifti_fp() -> bool:
        nifti_info, _ = generate_nifti_info()
        generated_nifti_fps = {
            nifti_info[nifti_idx]["fp"] for nifti_idx in range(len(nifti_info))
        }
        user_fp = Path("./tmp/session-data/user-options.json")
        with open(user_fp) as file:  # noqa: PTH123
            user_input = json.loads(file.read())
        output_dp = user_input["output_dcm_dp"]
        found_nifti_fps = set(
            glob.glob(os.path.join(output_dp, "**/*.nii"), recursive=True),  # noqa: PTH207, PTH118
        )
        return generated_nifti_fps == found_nifti_fps

    def check_nifti_integrity() -> bool:
        nifti_info, _ = generate_nifti_info()
        session_fp = Path("./tmp/session-data/clean/de-identified-files/session.json")
        with session_fp.open() as file:
            class_names = json.load(file)["classes"]
        class_idcs = set(range(len(class_names)))
        intensity_value_set = set()  # type: ignore[var-annotated]
        for nifti_ in nifti_info:
            nifti_fp = nifti_["fp"]
            intensity_value_set = set(
                np.unique(nib.load(nifti_fp).get_fdata().astype(np.uint8)),  # type: ignore[attr-defined]
            ).union(intensity_value_set)
        return intensity_value_set.issubset(class_idcs)

    homogeneity_state = check_class_names_integrity()
    if not homogeneity_state:
        renew_segmentation_data(["background"])
        return
    homogeneity_state = check_nifti_fp()
    if not homogeneity_state:
        renew_segmentation_data(["background"])
        return
    homogeneity_state = check_nifti_integrity()
    if not homogeneity_state:
        renew_segmentation_data(["background"])
        return


@app.post("/get_batch_classes")
async def get_batch_classes() -> list[str]:
    try:
        found_classes = get_classes_from_session()
    except AttributeError:
        sys.exit("E: Fatal Error; corrupted session file attribute detected")
    return found_classes


@app.post("/align_classes")
async def align_classes(classes: list[str]) -> None:
    renew_segmentation_data(classes)


@app.post("/custom_config/", name="custom_config")
async def custom_config(config_file: UploadFile) -> None:
    contents = await config_file.read()
    custom_fp = Path("./tmp/session-data/custom-config.csv")
    async with aiofiles.open(custom_fp, "wb") as file:
        await file.write(contents)


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
        mask=base64.b64encode(medsam_seg).decode("utf-8"),
        dimensions=[int(ws[inp_idx]), int(hs[inp_idx])],
    )


def prepare_medsam() -> None:
    medsam_model = load_model()
    raw_fp = Path("./tmp/session-data/raw")
    dcm_fps = sorted(raw_fp.glob("*"))
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
        cv2.polylines(bbox_mask, rectangle, isClosed=True, color=1, thickness=2)  # type: ignore[call-overload]
        cv2.fillPoly(multiplicative_mask, rectangle, 0)  # type: ignore[call-overload]
    multiplicative_mask = cv2.resize(  # type: ignore[assignment]
        multiplicative_mask,
        (initial_array_shape[1], initial_array_shape[0]),  # type: ignore[misc]
        interpolation=cv2.INTER_NEAREST,
    )
    bbox_mask = cv2.resize(  # type: ignore[assignment]
        bbox_mask,
        (initial_array_shape[1], initial_array_shape[0]),  # type: ignore[misc]
        interpolation=cv2.INTER_NEAREST,
    )
    additive_mask = reducted_region_color * (multiplicative_mask == 0)
    cleaned_img = img * multiplicative_mask + additive_mask
    bbox_img = np.maximum(bbox_mask * np.max(img), img)
    return cleaned_img, bbox_img  # type: ignore[return-value]


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
    return dcm, bbox_img  # type: ignore[return-value]


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


def adjust_dicom_metadata(  # noqa: C901, PLR0913
    dcm: pydicom.dataset.FileDataset,
    action_group_fp: str,
    patient_pseudo_id: str,
    days_total_offset: int,
    seconds_total_offset: int,
    patient_pseudo_id_prefix: str,
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
        patient_pseudo_id_prefix: str,
    ) -> pydicom.dataset.FileDataset:
        for ds_attr in ds:
            ds_tag_idx = re.sub("[(,) ]", "", str(ds_attr.tag)).upper()
            if ds[ds_tag_idx].VR == "SQ":
                for inner_ds_idx in range(ds[ds_tag_idx].VM):
                    ds[ds_tag_idx].value[inner_ds_idx] = recursive_sq_cleaner(
                        ds=ds[ds_tag_idx][inner_ds_idx],
                        action=action,
                        action_attr_tag_idx=action_attr_tag_idx,
                        patient_pseudo_id_prefix=patient_pseudo_id_prefix,
                    )
            elif action_attr_tag_idx == ds_tag_idx:
                if action == "Z":
                    # Check if ds_tag_idx is not one of the specified tags
                    if ds_tag_idx in ["00100010", "00100020"]:
                        ds[ds_tag_idx].value = (
                            patient_pseudo_id_prefix + patient_pseudo_id
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
                        tag_value_replacements["seconds_total_offset"] = (
                            seconds_total_offset
                        )
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
            patient_pseudo_id_prefix=patient_pseudo_id_prefix,
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
        self.out_dp = out_dp
        self.raw_data_dp = in_dp
        self.clean_data_dp = os.path.join(out_dp, "de-identified-files/")  # noqa: PTH118
        self.raw_dicom_paths = sorted(glob.glob(os.path.join(self.raw_data_dp, "*")))  # noqa: PTH207, PTH118
        self.clean_dicom_paths = sorted(
            glob.glob(os.path.join(self.clean_data_dp, "**/*.dcm"), recursive=True),  # noqa: PTH207, PTH118
        )
        self.dicom_pair_fps = []
        self.raw_dicom_hashes = []
        hashes_of_raw_files = []
        self.hashes_of_already_converted_files = []
        self.pending_deidentification = []
        for already_cleaned_dicom_path in self.clean_dicom_paths:
            self.hashes_of_already_converted_files.append(
                already_cleaned_dicom_path.split("/")[-1].split(".")[0],
            )
        for raw_dicom_fp in self.raw_dicom_paths:
            with open(file=raw_dicom_fp, mode="rb") as f:  # noqa: PTH123
                raw_dicom_bin = f.read()
                hashes_of_raw_files.append(hashlib.sha256(raw_dicom_bin).hexdigest())
        for hash_of_raw_file, raw_dicom_fp in zip(
            hashes_of_raw_files,
            self.raw_dicom_paths,
        ):
            if hash_of_raw_file in self.hashes_of_already_converted_files:
                clean_dicom_fp = self.clean_dicom_paths[
                    self.hashes_of_already_converted_files.index(hash_of_raw_file)
                ]
                self.dicom_pair_fps.append([raw_dicom_fp, clean_dicom_fp])
                self.pending_deidentification.append(False)
                self.raw_dicom_hashes.append(hash_of_raw_file)
            else:
                self.dicom_pair_fps.append([raw_dicom_fp, None])  # type: ignore[list-item]
                self.pending_deidentification.append(True)
                self.raw_dicom_hashes.append(hash_of_raw_file)
        for hash_of_already_converted_file, clean_dicom_fp in zip(
            self.hashes_of_already_converted_files,
            self.clean_dicom_paths,
        ):
            if hash_of_already_converted_file not in hashes_of_raw_files:
                self.dicom_pair_fps.append([clean_dicom_fp, clean_dicom_fp])
                self.pending_deidentification.append(False)
                self.raw_dicom_hashes.append(hash_of_already_converted_file)
        self.n_dicom_files = len(self.dicom_pair_fps)
        self.DICOM_IDX = -1

    def __next__(
        self,  # noqa: ANN101
    ) -> bool:
        self.DICOM_IDX += 1
        if self.n_dicom_files - 1 >= self.DICOM_IDX:
            self.raw_dicom_fp, self.clean_dicom_fp = self.dicom_pair_fps[self.DICOM_IDX]
            self.deintentify = self.pending_deidentification[self.DICOM_IDX]
            self.raw_dicom_hash = self.raw_dicom_hashes[self.DICOM_IDX]
            return True
        return False

    def define_undefined_clean_dicom_fp(self, clean_dcm) -> None:  # type: ignore[no-untyped-def] # noqa: ANN101, ANN001
        if self.deintentify:
            clean_dicom_dp = (
                self.clean_data_dp
                + str(clean_dcm[0x0010, 0x0020].value)
                + "/"
                + str(clean_dcm[0x0008, 0x0060].value)
                + "/"
                + str(clean_dcm[0x0020, 0x0011].value)
            )
            if not Path(clean_dicom_dp).exists():
                Path(clean_dicom_dp).mkdir(parents=True)
            clean_dicom_fp = os.path.join(clean_dicom_dp, self.raw_dicom_hash + ".dcm")  # noqa: PTH118
            self.dicom_pair_fps[self.DICOM_IDX][1] = self.clean_dicom_fp = (
                clean_dicom_fp
            )

    def export_processed_data(
        self,  # noqa: ANN101
        dcm: pydicom.dataset.FileDataset,
        bbox_img: NDArray[Any],
    ) -> None:
        if bbox_img is not None:
            bbox_img_fp = os.path.join(self.out_dp, self.raw_dicom_hash + "_bbox.png")  # noqa: PTH118
            Image.fromarray(bbox_img).save(bbox_img_fp)  # type: ignore[no-untyped-call]
            cache_bbox_img(self.raw_dicom_hash)
        dcm.save_as(self.clean_dicom_fp)

    def export_session(
        self,  # noqa: ANN101
        session: dict[str, dict[str, str]],
    ) -> None:
        session_fp = Path("./tmp/session-data/clean/de-identified-files/session.json")
        with session_fp.open("w") as file:
            json.dump(session, file, indent=4)


def dicom_deidentifier(
    session: dict,  # type: ignore[type-arg]
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
        filepath_or_buffer="./tmp/action-groups-dcm.csv",
        index_col=0,
    )
    with Path("./tmp/session-data/user-options.json").open() as file:
        user_input = json.load(file)
    pseudo_patient_ids = [
        int(patient_deidentification_properties["patient_pseudo_id"])
        for patient_deidentification_properties in session["de-identification"].values()
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
        dcm = pydicom.dcmread(rw_obj.raw_dicom_fp)
        bbox_img = None
        if rw_obj.deintentify:
            date_processing_choices = {"keep", "offset", "remove"}
            if user_input["date_processing"] not in date_processing_choices:
                msg = "E: Invalid date processing input"
                raise ValueError(msg)
            patient_pseudo_id_prefix = user_input["patient_pseudo_id_prefix"]
            real_patient_id = dcm[0x0010, 0x0020].value
            patient_deidentification_properties = session["de-identification"].get(
                real_patient_id,
                False,
            )
            if not patient_deidentification_properties:
                max_pseudo_patient_id += 1
                session["de-identification"][real_patient_id] = {
                    "patient_pseudo_id": "%.6d" % max_pseudo_patient_id,
                }
                days_total_offset = 10 * 365 + secrets.randbelow(1 + 10 * 365)
                seconds_total_offset = secrets.randbelow(24 * 60 * 60)
            else:
                days_total_offset = session["de-identification"][real_patient_id][
                    "days_offset"
                ]
                seconds_total_offset = session["de-identification"][real_patient_id][
                    "seconds_offset"
                ]
            dcm, tag_value_replacements = adjust_dicom_metadata(
                dcm=dcm,
                action_group_fp="./tmp/session-data/requested-action-group-dcm.csv",
                patient_pseudo_id=session["de-identification"][real_patient_id][
                    "patient_pseudo_id"
                ],
                days_total_offset=days_total_offset,
                seconds_total_offset=seconds_total_offset,
                patient_pseudo_id_prefix=patient_pseudo_id_prefix,
            )
            session["de-identification"][real_patient_id]["days_offset"] = (
                tag_value_replacements["days_total_offset"]
            )
            session["de-identification"][real_patient_id]["seconds_offset"] = (
                tag_value_replacements["seconds_total_offset"]
            )
            dcm = deidentification_attributes(user_input=user_input, dcm=dcm)
            if user_input["clean_image"]:
                dcm, bbox_img = image_deintentifier(dcm=dcm)  # type: ignore[assignment]
                bbox_img = image_preprocessing(  # type: ignore[assignment]
                    bbox_img,  # type: ignore[arg-type]
                    downscale_dimensionality=max(bbox_img.shape),  # type: ignore[attr-defined]
                    multichannel=True,
                    retain_aspect_ratio=True,
                )
            rw_obj.define_undefined_clean_dicom_fp(dcm)
            rw_obj.export_processed_data(dcm=dcm, bbox_img=bbox_img)  # type: ignore[arg-type]
        rw_obj.export_session(session=session)
    return session, rw_obj.dicom_pair_fps  # type: ignore[return-value]


@app.post("/submit_button")
async def handle_submit_button_click(user_options: UserOptionsClass) -> list[Any]:
    user_options = dict(user_options)  # type: ignore[assignment]
    user_options["input_dcm_dp"] = "./tmp/session-data/raw"  # type: ignore[index]
    user_options["output_dcm_dp"] = "./tmp/session-data/clean"  # type: ignore[index]
    user_fp = Path("./tmp/session-data/user-options.json")
    async with aiofiles.open(user_fp, "w") as file:
        await file.write(json.dumps(user_options, indent=4))
    session_filepath = "./tmp/session-data/clean/de-identified-files/session.json"
    if not Path(session_filepath).is_file():
        session = {"de-identification": {}, "classes": ["background"]}
    else:
        with Path(  # noqa: ASYNC101
            "./tmp/session-data/clean/de-identified-files/session.json",
        ).open() as file:
            session = json.load(file)
    session, dicom_pair_fps = dicom_deidentifier(  # type: ignore[assignment]
        session=session,
    )
    if user_options["annotation"]:  # type: ignore[index]
        prepare_medsam()
    return dicom_pair_fps


def generate_action_groups() -> None:
    threshold = 8
    acceptable_actions = {"", "K", "C", "Z", "X"}
    with request.urlopen(
        "https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_e.html",
    ) as response:
        html = response.read()
    df_table = pd.read_html(html)[3].fillna("")
    dicom_tags = df_table["Tag"].str.replace("[(,)]", "", regex=True).to_list()
    basic_profile = ["X" for i in df_table["Basic Prof."].to_list()]
    retain_safe_private_option = df_table["Rtn. Safe Priv. Opt."].to_list()
    retain_uids_option = df_table["Rtn. UIDs Opt."].to_list()
    retain_device_identity_option = df_table["Rtn. Dev. Id. Opt."].to_list()
    retain_patient_characteristics_option = [
        "K" if i == "C" else i for i in df_table["Rtn. Pat. Chars. Opt."].to_list()
    ]
    retain_long_modified_dates_option = df_table["Rtn. Long. Modif. Dates Opt."]
    retain_description_option = [
        "K" if i == "C" else i for i in df_table["Clean Desc. Opt."].to_list()
    ]
    output = [
        basic_profile,
        retain_safe_private_option,
        retain_uids_option,
        retain_device_identity_option,
        retain_patient_characteristics_option,
        retain_long_modified_dates_option,
        retain_description_option,
    ]
    dicom_tag_to_nema_action = dict(zip(dicom_tags, list(map(list, zip(*output)))))
    dicom_tag_to_nema_action_df = pd.DataFrame(dicom_tag_to_nema_action).transpose()
    dicom_tag_to_nema_action_df.columns = [
        "Default",
        "Rtn. Safe Priv. Opt.",
        "Rtn. UIDs Opt.",
        "Rtn. Dev. Id. Opt.",
        "Rtn. Pat. Chars. Opt.",
        "Rtn. Long. Modif. Dates Opt.",
        "Clean Desc. Opt.",
    ]
    dicom_tag_to_nema_action_df.insert(
        0,
        "Name",
        ["" for i in range(len(dicom_tag_to_nema_action_df))],
    )
    dicom_tag_to_nema_action_df.loc["00100010", "Default"] = "Z"
    dicom_tag_to_nema_action_df.loc["00100020", "Default"] = "Z"
    dicom_tag_to_nema_action_df = dicom_tag_to_nema_action_df.rename(
        columns={"Clean Desc. Opt.": "Rtn. Desc. Opt."},
    )
    dicom_tag_to_nema_action_df.insert(
        loc=6,
        column="Offset Long. Modif. Dates Opt.",
        value=dicom_tag_to_nema_action_df["Rtn. Long. Modif. Dates Opt."],
    )
    dicom_tag_to_nema_action_df.insert(
        loc=7,
        column="Remove Long. Modif. Dates Opt.",
        value=dicom_tag_to_nema_action_df["Rtn. Long. Modif. Dates Opt."],
    )
    dicom_tag_to_nema_action_df = dicom_tag_to_nema_action_df.replace(
        to_replace={"Rtn. Long. Modif. Dates Opt.": "C"},
        value="K",
    )
    dicom_tag_to_nema_action_df = dicom_tag_to_nema_action_df.replace(
        to_replace={"Remove Long. Modif. Dates Opt.": "C"},
        value="X",
    )
    dicom_tag_to_nema_action_df = dicom_tag_to_nema_action_df.replace(
        to_replace={
            "Rtn. Safe Priv. Opt.": "C",
            "Rtn. UIDs Opt.": "C",
            "Rtn. Dev. Id. Opt.": "C",
            "Rtn. Pat. Chars. Opt.": "C",
            "Rtn. Desc. Opt.": "C",
        },
        value="K",
    )
    for action in ("D", "X/Z", "X/Z/D", "X/D", "Z/D"):
        dicom_tag_to_nema_action_df = dicom_tag_to_nema_action_df.replace(
            to_replace=action,
            value="X",
        )
    for tag_idx in dicom_tag_to_nema_action_df.index:
        tag = "(" + tag_idx[0:4] + "," + tag_idx[4:9] + ")"
        if len(tag_idx) > threshold:
            if tag_idx != "ggggeeee where gggg is odd":
                msg = "Tag index error"
                raise ValueError(msg)
            tag = "(" + tag_idx[0:4] + "," + tag_idx[4:8] + ")" + tag_idx[8:]
        dicom_tag_to_nema_action_df.loc[tag_idx, "Name"] = df_table.loc[
            df_table["Tag"] == tag,
            "Attribute Name",
        ].item()
        if tag_idx in ["00100010", "00100020"]:
            dicom_tag_to_nema_action_df.loc[tag_idx, "Default"] = "Z"
        elif (
            "K" not in dicom_tag_to_nema_action_df.loc[tag_idx].to_numpy().tolist()[1:]
        ):
            dicom_tag_to_nema_action_df.loc[tag_idx, "Default"] = "X"
    dicom_tag_to_nema_action_df = dicom_tag_to_nema_action_df.sort_values(by="Name")

    patient_id_actions = dicom_tag_to_nema_action_df.loc["00100010"].to_numpy()[2:]
    patient_name_actions = dicom_tag_to_nema_action_df.loc["00100020"].to_numpy()[2:]
    if not (
        all(action == "" for action in patient_id_actions)
        and all(action == "" for action in patient_name_actions)
    ):
        msg = "Patient ID and name actions must remain empty except the default one"
        raise ValueError(
            msg,
        )

    dicom_tag_to_nema_action_arr = dicom_tag_to_nema_action_df.iloc[:, 1:].to_numpy()
    action_set = set(dicom_tag_to_nema_action_arr.flatten())
    if not action_set <= acceptable_actions:
        msg = "Unacceptable values found in action table"
        raise ValueError(msg)

    dicom_tag_to_nema_action_df.to_csv("./tmp/action-groups-dcm.csv")


def ndarray_size(arr: NDArray[Any]) -> int:
    return arr.itemsize * arr.size


def meddisc() -> None:
    tmp_directories = [
        Path("tmp"),
        Path("tmp/session-data/raw"),
        Path("tmp/session-data/clean"),
        Path("tmp/session-data/embed"),
    ]
    for directory in tmp_directories:
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = Path("tmp/action-groups-dcm.csv")
    vit_path = Path("tmp/tiny_vit_sam.py")
    vit_url = "https://api.github.com/repos/bowang-lab/MedSAM/contents/tiny_vit_sam.py?ref=LiteMedSAM"
    if not csv_path.exists():
        generate_action_groups()
    if not vit_path.exists():
        response = requests.get(vit_url, timeout=10)
        content = response.json()["content"]
        file_content = base64.b64decode(content).decode("utf-8")
        with vit_path.open("w") as f:
            f.write(file_content)
    if os.getenv("STAGING"):
        if not Path("tmp/fullchain.pem").exists():
            subprocess.run(
                [  # noqa: S603
                    "/usr/bin/openssl",
                    "req",
                    "-subj",
                    "/C=..",
                    "-nodes",
                    "-x509",
                    "-keyout",
                    "tmp/privkey.pem",
                    "-out",
                    "tmp/fullchain.pem",
                ],
                check=True,
            )
        run(
            app,
            host="0.0.0.0",  # noqa: S104
            port=8000,
            ssl_certfile="tmp/fullchain.pem",
            ssl_keyfile="tmp/privkey.pem",
        )
    else:
        results = pytest.main(["-rA", "-o", "cache_dir=tmp", __file__])
        if results.value != 0:  # type: ignore[attr-defined]
            sys.exit(results)


def main_cli() -> None:
    import fire

    fire.Fire(meddisc)


if __name__ == "__main__":
    meddisc()
