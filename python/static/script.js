var UploadForm = document.getElementById('UploadForm');
var SubmitAnonymizationProcess = document.getElementById('SubmitAnonymizationProcess');
var MetadataTable = document.getElementById('MetadataTable');
var DICOMOverview = document.getElementById('DICOMOverview');
var RawImg = document.getElementById('RawImg');
var CleanedImg = document.getElementById('CleanedImg');
var RawImgInner = document.getElementById('RawImgInner');
var CleanedImgInner = document.getElementById('CleanedImgInner');
var DICOMSlider = document.getElementById('DICOMSlider');
var ConversionResult = document.getElementById('ConversionResult');
var clean_image = document.getElementById('clean-image');
var OverlayCanvas = document.getElementById('OverlayCanvas')
var ctx = OverlayCanvas.getContext('2d');
var ToggleEdit = document.getElementById('ToggleEdit');
var BrushSizeSlider = document.getElementById('BrushSizeSlider');
var BrushSelect = document.getElementById('BrushSelect');
var LoadDICOM = document.getElementById('ResetDICOM');
var ModifyDICOM = document.getElementById('ModifyDICOM');
var Undo = document.getElementById('Undo');
var Redo = document.getElementById('Redo');
var Mode = document.getElementById('Mode');
var BoxCanvas = document.getElementById('BoxCanvas');
var bctx = BoxCanvas.getContext('2d');
var notificationMessage = document.getElementById("notification-message");
var notificationIcon = document.getElementById("notification-icon");
var notificationText = document.getElementById("notification-text");
var retain_safe_private_input_checkbox = document.getElementById('retain-safe-private-input-checkbox');
var retain_uids_input_checkbox = document.getElementById('retain-uids-input-checkbox');
var retain_device_identity_input_checkbox = document.getElementById('retain-device-identity-input-checkbox');
var retain_patient_characteristics_input_checkbox = document.getElementById('retain-patient-characteristics-input-checkbox');
var date_processing_select = document.getElementById('date-processing-select');
var retain_descriptors_input_checkbox = document.getElementById('retain-descriptors-input-checkbox');
var patient_pseudo_id_prefix_input_text = document.getElementById('patient-pseudo-id-prefix-input-text');
var UploadStatus = document.getElementById('UploadStatus');
var n_uploaded_files;
var dicom_pair_fps;
var OpenSequences = [];
var DiffEnabled = false;
var dcm_idx_;
var isEditing = false;
var currentBrush = 'background';
var brushSize = 25;
var isDrawing = false;
var lastX = 0;
var lastY = 0;
var undoStack = [];
var redoStack = [];
var editMode = 'brush';
var BoxStart = null;
var BoxEnd = null;
var progress_saved = true;
var notificationTimeout;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
const colorMap = {
    1: [255, 0, 0, 255],
    2: [0, 0, 255, 255],
    3: [0, 255, 0, 255],
    4: [255, 255, 0, 255],
    5: [255, 165, 0, 255],
    6: [255, 0, 255, 255],
    7: [0, 255, 255, 255],
    8: [128, 0, 128, 255],
    9: [255, 192, 203, 255],
    10: [128, 128, 128, 255],
};
const reverseColorMap = {
    '255,0,0,255': 1,
    '0,0,255,255': 2,
    '0,255,0,255': 3,
    '255,255,0,255': 4,
    '255,165,0,255': 5,
    '255,0,255,255': 6,
    '0,255,255,255': 7,
    '128,0,128,255': 8,
    '255,192,203,255': 9,
    '128,128,128,255': 10,
};
let classesMap = ["background"];
let predefinedClassesMap;
var slider_pending_update = false;
var pending_dcm_idx = 0;
var dicom_pair;
var total_altered_dicom_tags = '-';
var LoadingState = false;
var PredeterminedHeight = '37vw';
var masks = '';
var modal = document.querySelector('#modal');
var openModal = document.querySelector('#open-button');
var overrideMasks = document.querySelector('#overrideMasks');
var useBatchMasks = document.querySelector('#useBatchMasks');
var classes_submitted_state = false;

function ShowDiff(ToggleValue)
{
    DiffEnabled = ToggleValue;
    MetadataTable.innerHTML = table(dicom_pair['raw_dicom_metadata'], dicom_pair['cleaned_dicom_metadata'], DiffEnabled);
}

function HideSequence(SequenceID)
{
    const OpenSequencesStringified = OpenSequences.map(JSON.stringify);
    const FirstOccurence = OpenSequencesStringified.indexOf(JSON.stringify(SequenceID));
    let style;
    let expand_row_symbol;
    if (FirstOccurence == -1)
    {
        OpenSequences.push(SequenceID);
        style = 'block';
        expand_row_symbol = '-';
    }
    else
    {
        OpenSequences.splice(FirstOccurence, 1);
        style = 'none';
        expand_row_symbol = '+';
    }
    document.getElementById(JSON.stringify(SequenceID)).style.display = style;
    document.getElementById(JSON.stringify(SequenceID) + '_expand_row').innerHTML = expand_row_symbol;
}

function table(RawDCMMetadataObject, CleanedDCMMetadataObject, DiffEnabled)
{
    function RecursiveLevelBuild(MetadataTable, RawDCMMetadataObjectLvN, CleanedDCMMetadataObjectLvN, OffesetLeftMarginLvN, ParentNodeIdx, DiffEnabled)
    {
        let CurrentNodeIdx = ParentNodeIdx.slice();
        const indentation_block_unit = `<div class="cell-expand-row-margin"></div>`;
        let indentation_block;
        CurrentNodeIdx.push(-1);
        for (let tagID in RawDCMMetadataObjectLvN)
        {
            let tagID_part1 = tagID.substring(0, 4);
            let tagID_part2 = tagID.substring(4);
            let tagID_ = `(${tagID_part1},${tagID_part2})`;
            const vr = RawDCMMetadataObjectLvN[tagID].vr;
            const name = RawDCMMetadataObjectLvN[tagID].name;
            const raw_value = RawDCMMetadataObjectLvN[tagID].value;
            let cleaned_value;
            let tag_dropped;
            let right_row_contents;
            let left_col_style;
            let right_col_style;
            CurrentNodeIdx[CurrentNodeIdx.length - 1] += 1;
            if (CleanedDCMMetadataObjectLvN.hasOwnProperty(tagID))
            {
                cleaned_value = CleanedDCMMetadataObjectLvN[tagID].value;
                tag_dropped = false;
            }
            else
            {
                cleaned_value = '<TAG REMOVED>';
                tag_dropped = true;
            }
            if (tag_dropped)
            {
                left_col_style = ' style="background-color: rgba(0, 255, 255, 10%);"';
                right_col_style = ' style="background-color: rgba(47, 47, 51, 255);"';
                right_row_contents = 
                `
                `;
            }
            else
            {
                if (JSON.stringify(cleaned_value) !== JSON.stringify(raw_value))
                {
                    total_altered_dicom_tags += 1;
                    left_col_style = ' style="background-color: rgba(0, 255, 255, 10%);"';
                    right_col_style = left_col_style;
                }
                else
                {
                    left_col_style = ' style="background-color: rgba(100, 100, 110, 10%);"';
                    right_col_style = left_col_style;
                    if (DiffEnabled)
                    {
                        continue;
                    }
                }
                right_row_contents = 
                `
                    <div class="cell-dcmtag-id"${right_col_style}>${tagID_}</div>
                    <div class="cell-dcmtag-vr"${right_col_style}>${vr}</div>
                    <div class="cell-dcmtag-value"${right_col_style}>${cleaned_value}</div>
                    <div class="cell-dcmtag-name"${right_col_style}>${name}</div>
                `;
            }
            if (vr === 'SQ')
            {
                indentation_block = indentation_block_unit.repeat(OffesetLeftMarginLvN);
                MetadataTable += 
                `
                    <div class="outer-row" onclick="HideSequence(${JSON.stringify(CurrentNodeIdx)})" style="cursor: pointer;">
                        <div class="inner-row">
                            <div class="left-row">
                                ${indentation_block}
                                <div class="cell-expand-row" id="${JSON.stringify(CurrentNodeIdx)}_expand_row"${left_col_style}>+</div>
                                <div class="cell-dcmtag-id"${left_col_style}>${tagID_}</div>
                                <div class="cell-dcmtag-vr"${left_col_style}>${vr}</div>
                                <div class="cell-dcmtag-value"${left_col_style}></div>
                                <div class="cell-dcmtag-name"${left_col_style}>${name}</div>
                            </div>
                            <div class="cell-vertical-separator"></div>
                            <div class="right-row">
                                ${indentation_block}
                                <div class="cell-dcmtag-id"${right_col_style}>${tagID_}</div>
                                <div class="cell-dcmtag-vr"${right_col_style}>${vr}</div>
                                <div class="cell-dcmtag-value"${right_col_style}></div>
                                <div class="cell-dcmtag-name"${right_col_style}>${name}</div>
                            </div>
                        </div>
                    </div>
                `;
                MetadataTable = 
                `
                    ${MetadataTable}
                    <div id="${JSON.stringify(CurrentNodeIdx)}" style="display: none;">
                `;
                CurrentNodeIdx.push(-1);
                for (let ds_idx in raw_value)
                {
                    CurrentNodeIdx[CurrentNodeIdx.length - 1] += 1;
                    indentation_block = indentation_block_unit.repeat(OffesetLeftMarginLvN+1);
                    MetadataTable += 
                    `
                        <div class="outer-row">
                            <div class="inner-row">
                                <div class="left-row">
                                    ${indentation_block}
                                    <div class="cell-dataset">
                                        Dataset ${ds_idx}
                                    </div>
                                </div>
                                <div class="cell-vertical-separator"></div>
                                <div class="right-row">
                                    ${indentation_block}
                                    <div class="cell-dataset">
                                        Dataset ${ds_idx}
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    const RawDCMMetadataObjectLvN = raw_value[ds_idx];
                    const CleanedDCMMetadataObjectLvN = cleaned_value[ds_idx];
                    MetadataTable = RecursiveLevelBuild(MetadataTable, RawDCMMetadataObjectLvN, CleanedDCMMetadataObjectLvN, OffesetLeftMarginLvN+2, CurrentNodeIdx, DiffEnabled);
                }
                MetadataTable = 
                `
                        ${MetadataTable}
                    </div>
                `;
                CurrentNodeIdx = CurrentNodeIdx.slice(0, CurrentNodeIdx.length-1);
            }
            else
            {
                indentation_block = indentation_block_unit.repeat(OffesetLeftMarginLvN);
                MetadataTable += 
                `
                    <div class="outer-row">
                        <div class="inner-row">
                            <div class="left-row">
                                ${indentation_block}
                                <div class="cell-expand-row"${left_col_style}></div>
                                <div class="cell-dcmtag-id"${left_col_style}>${tagID_}</div>
                                <div class="cell-dcmtag-vr"${left_col_style}>${vr}</div>
                                <div class="cell-dcmtag-value"${left_col_style}>${raw_value}</div>
                                <div class="cell-dcmtag-name"${left_col_style}>${name}</div>
                            </div>
                            <div class="cell-vertical-separator"></div>
                            <div class="right-row">
                                ${indentation_block}
                                ${right_row_contents}
                            </div>
                        </div>
                    </div>
                `;
            }
        }
        return MetadataTable;
    };
    total_altered_dicom_tags = 0;
    let UppermostRowStyle = ` style="background-color: rgba(45, 45, 50, 255);"`;
    let MetadataTable = 
    `
        <div class="outer-row">
            <div class="inner-row-column-names">
                <div class="left-row">
                    <div class="cell-expand-row"></div>
                    <div class="cell-dcmtag-id"${UppermostRowStyle}><b>Tag ID</b></div>
                    <div class="cell-dcmtag-vr"${UppermostRowStyle}><b>VR</b></div>
                    <div class="cell-dcmtag-value"${UppermostRowStyle}><b>Tag Value</b></div>
                    <div class="cell-dcmtag-name"${UppermostRowStyle}><b>Tag Name</b></div>
                </div>
                <div class="cell-vertical-separator"></div>
                <div class="right-row">
                    <div class="cell-dcmtag-id"${UppermostRowStyle}><b>Tag ID</b></div>
                    <div class="cell-dcmtag-vr"${UppermostRowStyle}><b>VR</b></div>
                    <div class="cell-dcmtag-value"${UppermostRowStyle}><b>Tag Value</b></div>
                    <div class="cell-dcmtag-name"${UppermostRowStyle}><b>Tag Name</b></div>
                </div>
            </div>
        </div>
    `;
    const OffesetLeftMarginLv0 = 0;
    const ParentNodeIdx = [];
    const RawDCMMetadataObjectLv0 = RawDCMMetadataObject;
    const CleanedDCMMetadataObjectLv0 = CleanedDCMMetadataObject;
    MetadataTable = RecursiveLevelBuild(MetadataTable, RawDCMMetadataObjectLv0, CleanedDCMMetadataObjectLv0, OffesetLeftMarginLv0, ParentNodeIdx, DiffEnabled);
    return MetadataTable;
};

function CheckForChanges()
{
    if (slider_pending_update)
    {
        UpdateDICOMInformation(pending_dcm_idx);
    }
    setTimeout(CheckForChanges, 50);
};

async function UpdateDICOMInformation(dcm_idx)
{
    if (progress_saved == true)
    {
        slider_pending_update = true;
        pending_dcm_idx = dcm_idx;
        if (LoadingState)
        {
            return;
        }
        slider_pending_update = false;
        LoadingState = true;
        dcm_idx_ = dcm_idx
        const dicom_pair_fp = await dicom_pair_fps[dcm_idx_]
        const conversion_info_response = await fetch
        (
            '/conversion_info/',
            {
                method: 'POST',
                headers:
                {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(dicom_pair_fp)
            }
        );
        dicom_pair = await conversion_info_response.json();
        const dicom_metadata_table = table(dicom_pair['raw_dicom_metadata'], dicom_pair['cleaned_dicom_metadata'], DiffEnabled);
        if (RawImgInner.height !== 0)
        {
            PredeterminedHeight = String(RawImgInner.height) + 'px';
        }
        try
        {
            modality = dicom_pair['raw_dicom_metadata']['00080060'].value;
        }
        catch (error)
        {
            modality = '-';
        }
        if (classes_submitted_state)
        {
            await get_mask_from_file();
            undoStack = [];
            redoStack = [];
        }
        DICOMOverview.innerHTML =
        `
            Index: ${dcm_idx_}
            </br>
            Raw File Path: ${dicom_pair_fp[0]}
            </br>
            Clean File Path: ${dicom_pair_fp[1]}
            </br>
            Patient's Original ID: ${dicom_pair['raw_dicom_metadata']['00100020'].value}
            </br>
            Modality: ${modality}
            </br>
            Total number of altered tags (excluding the pixel data): ${total_altered_dicom_tags}
        `;
        MetadataTable.innerHTML = dicom_metadata_table;
        RawImg.style.minHeight = PredeterminedHeight;
        CleanedImg.style.minHeight = PredeterminedHeight;
        RawImgInner.src = `data:image/png;base64,${dicom_pair['raw_dicom_img_data']}`;
        CleanedImgInner.src = `data:image/png;base64,${dicom_pair['cleaned_dicom_img_data']}`;
        RawImg.style.minHeight = 0;
        CleanedImg.style.minHeight = 0;
        LoadingState = false;
    }
    else
    {
        save = confirm('Unsaved changes found. Save last changes before continuing?')
        if (save)
        {
            await modify_dicom()
            await UpdateDICOMInformation(dcm_idx)
        }
        else
        {
            progress_saved = true
            await UpdateDICOMInformation(dcm_idx)
        }
    }
}

function base64torgba(encodedData) {
    const binaryString = window.atob(encodedData);
    const len = binaryString.length;
    const bytes = new Uint8ClampedArray(len * 4);
    for (let i = 0, j = 0; i < len; i++, j += 4) {
        let pixelValue = binaryString.charCodeAt(i);
        const color = colorMap[pixelValue];
        if (color) {
            bytes.set(color, j);
        } else {
            bytes.set([0, 0, 0, 0], j);
        }
    }
    return bytes
}

document.querySelector('#UploadForm input[name="files"]').addEventListener
(
    'change',
    async function()
    {
        const formData = new FormData();
        for (let i = 0; i < this.files.length; i++)
        {
            formData.append('files', this.files[i]);
        }
        const dcm_files_response = await fetch
        (
            '/upload_files/',
            {
                method: 'POST',
                body: formData
            }
        );
        const dcm_files = await dcm_files_response.json();
        ConversionResult.style.display = 'none';
        if (dcm_files_response.ok && dcm_files.n_uploaded_files > 0)
        {
            n_uploaded_files = dcm_files.n_uploaded_files;
            UploadStatus.innerHTML = 
            `
                </br>\n
                Files Uploaded Successfully\n
                </br>\n
                Total DICOM files: ${n_uploaded_files}\n
                </br>\n
                Total size of DICOM content: ${dcm_files.total_size} MB\n
                </br>\n
                </br>
            `;
            SubmitAnonymizationProcess.disabled = false;
            resetGUIElements();
        }
        else
        {
            alert('W: Invalid directory input. Make sure it contains at least one DICOM file')
        }
    }
);

document.querySelector('#SessionForm input[name="SessionFile"]').addEventListener
(
    'change',
    async function(e)
    {
        e.preventDefault();
        const fileInput = this;
        const file = fileInput.files[0];
        const reader = new FileReader();
        reader.onload = async function()
        {
            const payload = JSON.parse(reader.result);
            const response = await fetch
            (
                '/session/',
                {
                    method: 'POST',
                    headers:
                    {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                }
            );
        };
        reader.readAsText(file);
    }
);

window.onload = function()
{
    document.querySelector('.UploadConfig').addEventListener
    (
        'change',
        function(e)
        {
            e.preventDefault();
            if (this.value) {
                const form = this.form;
                const formData = new FormData(form);
                fetch
                (
                    form.action,
                    {
                        method: form.method,
                        body: formData
                    }
                )
                retain_safe_private_input_checkbox.checked = false;
                retain_uids_input_checkbox.checked = false;
                retain_device_identity_input_checkbox.checked = false;
                retain_patient_characteristics_input_checkbox.checked = false;
                date_processing_select.value = 'offset';
                retain_descriptors_input_checkbox.checked = false;
                retain_safe_private_input_checkbox.disabled = true;
                retain_uids_input_checkbox.disabled = true;
                retain_device_identity_input_checkbox.disabled = true;
                retain_patient_characteristics_input_checkbox.disabled = true;
                date_processing_select.disabled = true;
                retain_descriptors_input_checkbox.disabled = true;
            }
        }
    );
};

async function submit_dicom_processing_request()
{
    SubmitAnonymizationProcess.disabled = true;
    const data =
    {
        'clean_image': clean_image.checked,
        'retain_safe_private': retain_safe_private_input_checkbox.checked,
        'retain_uids': retain_uids_input_checkbox.checked,
        'retain_device_identity': retain_device_identity_input_checkbox.checked,
        'retain_patient_characteristics': retain_patient_characteristics_input_checkbox.checked,
        'date_processing': date_processing_select.value,
        'retain_descriptors': retain_descriptors_input_checkbox.checked,
        'patient_pseudo_id_prefix': patient_pseudo_id_prefix_input_text.value
    };
    const dicom_pair_fps_response = await fetch
    (
        '/submit_button',
        {
            method: 'POST',
            headers:
            {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        }
    );
    dicom_pair_fps = await dicom_pair_fps_response.json();
    DICOMSlider.max = n_uploaded_files-1;
    DICOMSlider.value = 0;
    await UpdateDICOMInformation(0);
    CheckForChanges();
    ConversionResult.style.display = 'inline';
    retain_safe_private_input_checkbox.disabled = false;
    retain_uids_input_checkbox.disabled = false;
    retain_device_identity_input_checkbox.disabled = false;
    retain_patient_characteristics_input_checkbox.disabled = false;
    date_processing_select.disabled = false;
    retain_descriptors_input_checkbox.disabled = false;
    await fetch
    (
        '/correct_seg_homogeneity',
        {
            method: 'POST'
        }
    );
    const predefinedClassesMap_responce = await fetch
    (
        '/get_batch_classes',
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        }
    );
    predefinedClassesMap = await predefinedClassesMap_responce.json()
    predefinedClassesMap = predefinedClassesMap.classes
    classesMap = Array.from(predefinedClassesMap)
    for (let class_idx = 1; class_idx < classesMap.length; class_idx++)
    {
        const newOption = new Option(classesMap[class_idx], classesMap[class_idx], false, false);
        BrushSelect.add(newOption);
    }
}

async function get_mask_from_file() {
    const reset_response = await fetch(
        '/get_mask_from_file/',
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(dicom_pair_fps[dcm_idx_][1])
        });
    if (reset_response.ok) {
        const response_data = await reset_response.json();
        fillCanvas(response_data['PixelData'], response_data['dimensions']);
        showNotification("success", "Loaded mask from DICOM", 1500);
    }
}

async function modify_dicom() {
    const requestBody = {
        pixelData: canvastobase64(),
        filepath: dicom_pair_fps[dcm_idx_][1],
        classes: classesMap
    }; 
    const modify_response = await fetch(
        '/modify_dicom/',
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
    if (modify_response.ok) {
        progress_saved = true
        showNotification("success", "Saved to DICOM", 1500);
    }
}

function canvastobase64() {
    const canvasData = ctx.getImageData(0, 0, OverlayCanvas.width, OverlayCanvas.height);
    const data = canvasData.data;
    let binaryString = '';
    for (let i = 0; i < data.length; i += 4) {
        let rgba = `${data[i]},${data[i + 1]},${data[i + 2]},${data[i + 3]}`;
        if (data[i + 3] === 0) {
            binaryString += String.fromCharCode(0);
        } else if (rgba in reverseColorMap) {
            binaryString += String.fromCharCode(reverseColorMap[rgba]);
        } else {
            binaryString += String.fromCharCode(0);
        }
    }
    return window.btoa(binaryString);
}

BrushSelect.addEventListener('change', (event) => {
    currentBrush = event.target.value;
    updateBrushIndicator(classesMap.indexOf(currentBrush));
});

BrushSizeSlider.addEventListener('input', (event) => {
    brushSize = event.target.value;
});

ToggleEdit.addEventListener('click', () => {
    isEditing = !isEditing;
    ToggleEdit.textContent = isEditing ? 'Edit Mode' : 'View Mode';
    OverlayCanvas.style.pointerEvents = isEditing ? 'auto' : 'none';
    BoxCanvas.style.pointerEvents = (isEditing && editMode === 'boundingBox') ? 'auto' : 'none';
    showNotification("info", "Switched to " + ToggleEdit.textContent, 1500);
});

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;
    return {
        x: (evt.clientX - rect.left) * scaleX,
        y: (evt.clientY - rect.top) * scaleY
    };
}

function draw(e) {
    if (!isEditing) return;
    var mousePos = getMousePos(OverlayCanvas, e);
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const brushClassNumber = classesMap.indexOf(currentBrush);
    const brushColor = brushClassNumber !== -1 && colorMap[brushClassNumber] 
                        ? `rgba(${colorMap[brushClassNumber].join(',')})` 
                        : 'rgba(0,0,0,1)';
    ctx.strokeStyle = brushColor;
    ctx.globalCompositeOperation = currentBrush === 'background' ? 'destination-out' : 'source-over';
    if (!isDrawing) return;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(mousePos.x, mousePos.y);
    ctx.stroke();
    lastX = mousePos.x;
    lastY = mousePos.y;
}

function saveState() {
    undoStack.push(ctx.getImageData(0, 0, OverlayCanvas.width, OverlayCanvas.height));
    redoStack = [];
}

function undoLastAction() {
    if (undoStack.length > 0) {
        redoStack.push(undoStack.pop());
        if (undoStack.length > 0) {
            const previousState = undoStack[undoStack.length - 1];
            ctx.putImageData(previousState, 0, 0);
        } else {
            ctx.clearRect(0, 0, OverlayCanvas.width, OverlayCanvas.height);
        }
    }
    showNotification("info", "Undo", 1500);
}

function redoLastAction() {
    if (redoStack.length > 0) {
        const nextState = redoStack.pop();
        undoStack.push(nextState);
        ctx.putImageData(nextState, 0, 0);
    }
    showNotification("info", "Redo", 1500);
}

document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        undoLastAction();
    }
    if ((e.ctrlKey && e.key === 'y') || (e.metaKey && e.shiftKey && e.key === 'Z')) {
        e.preventDefault();
        redoLastAction();
    }
});
Undo.addEventListener('click', undoLastAction);
Redo.addEventListener('click', redoLastAction);

OverlayCanvas.addEventListener('mousedown', (e) => {
    var mousePos = getMousePos(OverlayCanvas, e);
    isDrawing = true;
    [lastX, lastY] = [mousePos.x, mousePos.y];
});
OverlayCanvas.addEventListener('mousemove', draw);
OverlayCanvas.addEventListener('mouseup', () => {
    saveState();
    isDrawing = false;
});
OverlayCanvas.addEventListener('mouseout', () => isDrawing = false);

function clearBoundingBox() {
    bctx.clearRect(0, 0, BoxCanvas.width, BoxCanvas.height);
}

async function medsam_estimation(normalizedStart,normalizedEnd) {
    if (editMode !== 'boundingBox' || !BoxStart || !BoxEnd) return;
    const boxRequest = {
        normalizedStart: normalizedStart,
        normalizedEnd: normalizedEnd,
        segClass: classesMap.indexOf(BrushSelect.value),
        inpIdx: dcm_idx_
    };
    const box_response = await fetch(
        '/medsam_estimation/',
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(boxRequest)
        });
    if (box_response.ok) {
        const box_data = await box_response.json();
        const dims = box_data['dimensions']
        mergeMask(ctx, box_data['mask'], dims[0], dims[1], colorMap);
        saveState();
    }
}

Mode.addEventListener('click', function () {
    if (editMode === 'brush') {
        editMode = 'boundingBox';
        Mode.textContent = 'Box';
        BoxCanvas.style.pointerEvents = isEditing ? 'auto' : 'none';
        document.querySelector('#BrushSelect option[value="background"]').disabled = true;
        let foundBackground = false;
        let firstAvailableOption = null;
        document.querySelectorAll('#BrushSelect option').forEach(option => {
            if (foundBackground && !option.disabled) {
                firstAvailableOption = firstAvailableOption || option;
            }
            if (option.value === 'background') {
                foundBackground = true;
            }
        });
        if (currentBrush=='background') {
            BrushSelect.value = firstAvailableOption.value;
            BrushSelect.dispatchEvent(new Event('change'));
        }
    } else {
        editMode = 'brush';
        Mode.textContent = 'Brush';
        BoxCanvas.style.pointerEvents = 'none';
        document.querySelector('#BrushSelect option[value="background"]').disabled = false;
    }
    showNotification("info", "Switched to " + Mode.textContent, 1500);
});

BoxCanvas.addEventListener('mousedown', (e) => {
    if (editMode !== 'boundingBox') return;
    var mousePos = getMousePos(BoxCanvas, e);
    BoxStart = mousePos;
    BoxEnd = null;
});

BoxCanvas.addEventListener('mousemove', (e) => {
    if (editMode !== 'boundingBox' || !BoxStart) return;
    var mousePos = getMousePos(BoxCanvas, e);
    BoxEnd = mousePos;
    clearBoundingBox();
    bctx.beginPath();
    bctx.lineWidth = 5; 
    bctx.rect(BoxStart.x, BoxStart.y, BoxEnd.x - BoxStart.x, BoxEnd.y - BoxStart.y);
    const classNumber = classesMap.indexOf(currentBrush);
    bctx.strokeStyle = `rgba(${colorMap[classNumber].join(',')})`;
    bctx.stroke();
});

BoxCanvas.addEventListener('mouseup', () => {
    if (editMode !== 'boundingBox' || !BoxStart || !BoxEnd) return;
    const normalizedStart = {
        x: BoxStart.x / BoxCanvas.width,
        y: BoxStart.y / BoxCanvas.height
    };
    const normalizedEnd = {
        x: BoxEnd.x / BoxCanvas.width,
        y: BoxEnd.y / BoxCanvas.height
    };  
    medsam_estimation(normalizedStart,normalizedEnd);
    clearBoundingBox();
    BoxStart = null;
});

BoxCanvas.addEventListener('mouseout', () => {
    if (editMode === 'boundingBox') clearBoundingBox();
    BoxStart = null;
});

function fillCanvas(maskData, dimensions) {
    OverlayCanvas.width = dimensions[0];
    OverlayCanvas.height = dimensions[1];
    BoxCanvas.width = dimensions[0];
    BoxCanvas.height = dimensions[1];
    const drawData = new ImageData(base64torgba(maskData), dimensions[0], dimensions[1]);
    ctx.putImageData(drawData, 0, 0);
}

function add_class() {
    const inputVal = ClassText.value.trim();
    if (inputVal === '') {
        showNotification("info", "Please enter a class name", 1500);
        return;
    }
    if (classesMap.includes(inputVal)) {
        showNotification("failure", "This class already exists", 1500);
        return;
    }
    if (classesMap.length >= 11) {
        showNotification("failure", "Maximum of 10 classes reached", 1500);
        return;
    }
    classesMap.push(inputVal);
    const newOption = new Option(inputVal, inputVal, false, true);
    BrushSelect.add(newOption);
    BrushSelect.value = inputVal;
    ClassText.value = '';
    const event = new Event('change');
    BrushSelect.dispatchEvent(event);
    showNotification("success", "Added class " + inputVal, 1500);
}

function remove_class() {
    const inputVal = ClassText.value.trim();
    if (inputVal === '') {
        showNotification("info", "Please enter a class name", 1500);
        return;
    }
    const index = classesMap.indexOf(inputVal);
    if (index === -1) {
        showNotification("failure", "Class not found", 1500);
        return;
    }
    if (inputVal === 'background') {
        showNotification("failure", "Cannot remove background class", 1500);
        return;
    }
    classesMap.splice(index, 1);
    for (let i = 0; i < BrushSelect.options.length; i++) {
        if (BrushSelect.options[i].value === inputVal) {
            BrushSelect.remove(i);
            break;
        }
    }
    ClassText.value = '';
    const event = new Event('change');
    BrushSelect.dispatchEvent(event);
    showNotification("success", "Removed class " + inputVal, 1500);
}

async function submit_classes(){
    classes_submitted_state = true;
    ToggleEdit.disabled = false;
    if (classesMap.length > 1){
        Mode.disabled = false;
    }
    BrushSizeSlider.disabled = false;
    Undo.disabled = false;
    Redo.disabled = false;
    LoadDICOM.disabled = false;
    ModifyDICOM.disabled = false;
    Add.disabled = true;
    Remove.disabled = true;
    ClassText.disabled = true;
    SubmitClasses.disabled = true;
    if (predefinedClassesMap.length === 1 && predefinedClassesMap[0] == 'background')
    {
        get_mask_from_file();
    }
    else if (classesMap.length !== predefinedClassesMap.length)
    {
        modal.showModal();
    }
    else
    {
        for (let i = 0; i < classesMap.length; i++)
        {
            if (classesMap[i] !== predefinedClassesMap[i])
            {
                modal.showModal();
                break;
            }
        }
    }
    showNotification("success", "Submitted classes", 1500);
}

function mergeMask(ctx, base64DicomMask, canvasWidth, canvasHeight, colorMap) {
    const binaryString = window.atob(base64DicomMask);
    let imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    let data = imageData.data;
    for (let i = 0; i < binaryString.length; i++) {
        let maskValue = binaryString.charCodeAt(i);
        let index = i * 4;

        if (maskValue in colorMap) {
            data[index] = colorMap[maskValue][0];
            data[index + 1] = colorMap[maskValue][1];
            data[index + 2] = colorMap[maskValue][2];
            data[index + 3] = colorMap[maskValue][3];
        }
    }
    ctx.putImageData(imageData, 0, 0);
}

overrideMasks.addEventListener('click', async function(){
    await fetch
    (
        '/align_classes/',
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(classesMap)
        }
    );
    get_mask_from_file();
    modal.close();
})

useBatchMasks.addEventListener('click', function(){
    for (let class_idx = 1; class_idx < classesMap.length; class_idx++)
    {
        BrushSelect.remove(1);
    }
    classesMap = Array.from(predefinedClassesMap)
    for (let class_idx = 1; class_idx < classesMap.length; class_idx++)
    {
        const newOption = new Option(classesMap[class_idx], classesMap[class_idx], false, false);
        BrushSelect.add(newOption);
    }
    get_mask_from_file();
    modal.close();
})

function mergeMask(ctx, base64DicomMask, canvasWidth, canvasHeight, colorMap) {
    const binaryString = window.atob(base64DicomMask);
    let imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    let data = imageData.data;
    for (let i = 0; i < binaryString.length; i++) {
        let maskValue = binaryString.charCodeAt(i);
        let index = i * 4;

        if (maskValue in colorMap) {
            data[index] = colorMap[maskValue][0];
            data[index + 1] = colorMap[maskValue][1];
            data[index + 2] = colorMap[maskValue][2];
            data[index + 3] = colorMap[maskValue][3];
        }
    }
    progress_saved = false
    ctx.putImageData(imageData, 0, 0);
}

function updateBrushIndicator(brushNumber) {
    const colorIndicator = document.querySelector('.colorIndicator');
    if (brushNumber === 0) {
        colorIndicator.style.backgroundImage = 
            "linear-gradient(45deg, #808080 25%, transparent 25%, transparent 75%, #808080 75%)," +
            "linear-gradient(45deg, #808080 25%, #f0f0f0 25%, #f0f0f0 75%, #808080 75%)";
        colorIndicator.style.backgroundPosition = "0 0, 9px 9px";
        colorIndicator.style.backgroundSize = "18px 18px";
    } else {
        const color = colorMap[brushNumber];
        if (color) {
            const rgbaColor = `rgba(${color.join(',')})`;
            colorIndicator.style.backgroundColor = rgbaColor;

            colorIndicator.style.backgroundImage = "none";
        }
    }
}

function resetGUIElements() {
    for (var i = BrushSelect.options.length - 1; i >= 0; i--) {
        if (BrushSelect.options[i].value !== 'background') {
            BrushSelect.remove(i);
        }
    }
    BrushSelect.value = "background";
    const event = new Event('change');
    BrushSelect.dispatchEvent(event);
    classesMap = ["background"];
    classes_submitted_state = false;
    ToggleEdit.disabled = true;
    isEditing = false;
    ToggleEdit.textContent = 'View Mode';
    OverlayCanvas.style.pointerEvents = 'none';
    BoxCanvas.style.pointerEvents = 'none';
    Mode.disabled = true;
    editMode = 'brush';
    Mode.textContent = 'Brush';
    document.querySelector('#BrushSelect option[value="background"]').disabled = false;
    BrushSizeSlider.disabled = true;
    Undo.disabled = true;
    Redo.disabled = true;
    LoadDICOM.disabled = true;
    ModifyDICOM.disabled = true;
    Add.disabled = false;
    Remove.disabled = false;
    ClassText.disabled = false;
    SubmitClasses.disabled = false;
}

function showNotification(type, text, duration) {
    clearTimeout(notificationTimeout);
    notificationText.textContent = text;
    switch (type) {
        case "success":
            notificationIcon.textContent = "✔️"; 
            notificationMessage.style.backgroundColor = "green";
            notificationMessage.style.color = "white";
            break;
        case "info":
            notificationIcon.textContent = "ℹ️"; 
            notificationMessage.style.backgroundColor = "lightblue";
            notificationMessage.style.color = "black";
            break;
        case "failure":
            notificationIcon.textContent = "❌"; 
            notificationMessage.style.backgroundColor = "red";
            notificationMessage.style.color = "white";
            break;
        default:
            notificationIcon.textContent = "";
            notificationMessage.style.backgroundColor = "grey";
            notificationMessage.style.color = "black";
            break;
    }
    notificationMessage.style.display = "flex";
    notificationTimeout = setTimeout(function() {
        notificationMessage.style.display = "none";
    }, duration);
}