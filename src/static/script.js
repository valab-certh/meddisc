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
var LoadDICOM = document.getElementById('LoadDICOM');
var ModifyDICOM = document.getElementById('ModifyDICOM');
var Undo = document.getElementById('Undo');
var Redo = document.getElementById('Redo');
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

// GUI variables initialization 
var isEditing = false;
var currentBrush = 'eraser';
var brushSize = 25;
var isDrawing = false;
var lastX = 0;
var lastY = 0;
var undoStack = [];
var redoStack = [];

ctx.lineJoin = 'round';
ctx.lineCap = 'round';

// ! Loading state (1/4): Begin

var slider_pending_update = false;
var pending_dcm_idx = 0;
var dicom_pair;
var total_altered_dicom_tags = '-';

// Description
//     Clean solution to the issue of delayed slider responses. The advantage of this method compared to methods like throttling is that it continues execution as soon as the content is loaded into the HTML.
var LoadingState = false;

// ! Loading state (1/4): End

// Used for UX performance during the slider's image transitions. Helps (but is not sufficient by itself) for the prevention of that momentary flickering where during that time interval, the entire table below fills the empty space.
var PredeterminedHeight = '37vw';

var masks = '';


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
        // Expand sequence
        OpenSequences.push(SequenceID);
        style = 'block';
        expand_row_symbol = '-';
    }
    else
    {
        // Contract sequence
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

// Loading state (2/4)
function CheckForChanges()
{
    if (slider_pending_update)
    {
        UpdateDICOMInformation(pending_dcm_idx)
    }

    setTimeout(CheckForChanges, 50);
};

async function UpdateDICOMInformation(dcm_idx)
{
    // ! Loading state (3/4): Begin

    slider_pending_update = true;
    pending_dcm_idx = dcm_idx;

    if (LoadingState)
    {
        return;
    }

    slider_pending_update = false;

    LoadingState = true;

    // ! Loading state (3/4): End

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

    if (conversion_info_response.ok)
    {
        dicom_pair = await conversion_info_response.json();
        const dicom_metadata_table = table(dicom_pair['raw_dicom_metadata'], dicom_pair['cleaned_dicom_metadata'], DiffEnabled);
        const raw_dicom_img_fp = dicom_pair['raw_dicom_img_fp'];
        const cleaned_dicom_img_fp = dicom_pair['cleaned_dicom_img_fp'];
        const segmentation_data = dicom_pair['segmentation_data'];
        const dimensions = dicom_pair['dimensions']

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
        RawImgInner.src = raw_dicom_img_fp;
        CleanedImgInner.src = cleaned_dicom_img_fp;
        // fill canvas with segmentation mask
        OverlayCanvas.width = dimensions[0];
        OverlayCanvas.height = dimensions[1];
        const imageData = new ImageData(base64torgba(segmentation_data), dimensions[0], dimensions[1]);
        ctx.putImageData(imageData, 0, 0);
        RawImg.style.minHeight = 0;
        CleanedImg.style.minHeight = 0;
    }

    // Loading state (4/4)
    LoadingState = false;
}

function base64torgba(encodedData) {
    // decode base64
    const binaryString = window.atob(encodedData);
    const len = binaryString.length;
    const bytes = new Uint8ClampedArray(len * 4); // *4 for RGBA

    for (let i = 0, j = 0; i < len; i++, j += 4) {
        let pixelValue = binaryString.charCodeAt(i);
        if (pixelValue === 0) {
            // transparent
            bytes.set([0, 0, 0, 0], j);
        } else if (pixelValue === 1) {
            // red
            bytes.set([255, 0, 0, 255], j);
        } else if (pixelValue === 2) {
            // blue
            bytes.set([0, 0, 255, 255], j);
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

    await fetch
    (
        '/correct_segmentation_sequence/',
        {
            method: 'POST'
        }
    );

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

    // Builds slider based on number of converted input DICOM files
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
}

async function reset_mask() {
    const reset_response = await fetch(
        '/reset_mask/',
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(dicom_pair_fps[dcm_idx_][1])
        });
    
    if (reset_response.ok) {
        const response_data = await reset_response.json();
        const PixelData = response_data['PixelData'];
        const reset_dimensions = response_data['dimensions']
        OverlayCanvas.width = reset_dimensions[0];
        OverlayCanvas.height = reset_dimensions[1];
        const resetData = new ImageData(base64torgba(PixelData), reset_dimensions[0], reset_dimensions[1]);
        ctx.putImageData(resetData, 0, 0);
    }
}

async function modify_dicom() {   
    const requestBody = {
        pixelData: canvastobase64(),
        filepath: dicom_pair_fps[dcm_idx_][1]
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
        
    }
}


function canvastobase64() {
    const canvasData = ctx.getImageData(0, 0, OverlayCanvas.width, OverlayCanvas.height);
    const data = canvasData.data; // RGBA data
    let binaryString = '';

    for (let i = 0; i < data.length; i += 4) {
        let r = data[i];
        let g = data[i + 1];
        let b = data[i + 2];
        let a = data[i + 3];

        if (a === 0) {
            // transparent
            binaryString += String.fromCharCode(0);
        } else if (r === 255 && g === 0 && b === 0 && a === 255) {
            // red
            binaryString += String.fromCharCode(1);
        } else if (r === 0 && g === 0 && b === 255 && a === 255) {
            // blue
            binaryString += String.fromCharCode(2);
        } else {
            // default
            binaryString += String.fromCharCode(0); 
        }
    }
    return window.btoa(binaryString);
}

// GUI functions and listeners
// brush selection
BrushSelect.addEventListener('change', (event) => {
    currentBrush = event.target.value;
});

// brush radius selection
BrushSizeSlider.addEventListener('input', (event) => {
    brushSize = event.target.value;
});

// toggle edit
ToggleEdit.addEventListener('click', () => {
    isEditing = !isEditing;
    ToggleEdit.textContent = isEditing ? 'Edit Mode' : 'View Mode';
    OverlayCanvas.style.pointerEvents = isEditing ? 'auto' : 'none';
});

// mouse position function for scaling
function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    // calculate scaling factor
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;

    // adjust coordinates
    return {
        x: (evt.clientX - rect.left) * scaleX, // Adjusting X coordinate
        y: (evt.clientY - rect.top) * scaleY  // Adjusting Y coordinate
    };
}

// draw function
function draw(e) {
    if (!isEditing) return;

    var mousePos = getMousePos(OverlayCanvas, e);

    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (currentBrush === 'eraser') {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.strokeStyle = 'rgba(0,0,0,1)';
    } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = currentBrush === 'class1' ? 'rgba(255,0,0,1)' : 'rgba(0,0,255,1)';
    }

    if (!isDrawing) return;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(mousePos.x, mousePos.y);
    ctx.stroke();

    lastX = mousePos.x;
    lastY = mousePos.y;
}

// save state function
function saveState() {
    undoStack.push(ctx.getImageData(0, 0, OverlayCanvas.width, OverlayCanvas.height));
    redoStack = [];
}

// undo function
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
}

// redo function
function redoLastAction() {
    if (redoStack.length > 0) {
        const nextState = redoStack.pop();
        undoStack.push(nextState);
        ctx.putImageData(nextState, 0, 0);
    }
}

// event listeners for undo and redo
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
    saveState(); // save state after every stroke
    isDrawing = false;
});
OverlayCanvas.addEventListener('mouseout', () => isDrawing = false);