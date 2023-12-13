var form = document.getElementById('UploadForm');
var button = document.getElementById('SubmitAnonymizationProcess');
var slider = document.getElementById("DICOMRange");
var n_uploaded_files;
var dicom_pair_fps;
var OpenSequences = [];
var DiffEnabled = false;
var dcm_idx_;
var dicom_pair;
var total_altered_dicom_tags = '-';

// Used for the UX performance during the slider's image transitions. Prevents that momentary flickering where the entire table below fills the empty space.
let PredeterminedHeight = '37vw';


function ShowDiff(ToggleValue)
{
    DiffEnabled = ToggleValue;
    document.getElementById('MetadataTable').innerHTML = table(dicom_pair['raw_dicom_metadata'], dicom_pair['cleaned_dicom_metadata'], DiffEnabled)
}

function HideSequence(SequenceID)
{
    const OpenSequencesStringified = OpenSequences.map(JSON.stringify);
    const FirstOccurence = OpenSequencesStringified.indexOf(JSON.stringify(SequenceID))
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

    document.getElementById(JSON.stringify(SequenceID)).style.display = style
    document.getElementById(JSON.stringify(SequenceID) + '_expand_row').innerHTML = expand_row_symbol
}

function table(RawDCMMetadataObject, CleanedDCMMetadataObject, DiffEnabled)
{
    function RecursiveLevelBuild(MetadataTable, RawDCMMetadataObjectLvN, CleanedDCMMetadataObjectLvN, OffesetLeftMarginLvN, ParentNodeIdx, DiffEnabled)
    {
        let CurrentNodeIdx = ParentNodeIdx.slice();
        const indentation_block_unit = `<div class="cell-expand-row-margin"></div>`;
        let indentation_block;

        CurrentNodeIdx.push(-1)

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
                    total_altered_dicom_tags += 1
                    left_col_style = ' style="background-color: rgba(0, 255, 255, 10%);"';
                    right_col_style = left_col_style
                }
                else
                {
                    left_col_style = ' style="background-color: rgba(50, 50, 55, 255);"';
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

                CurrentNodeIdx.push(-1)

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

    total_altered_dicom_tags = 0

    let UppermostRowStyle = ` style="background-color: rgba(45, 45, 50, 255);"`;

    let MetadataTable = 
    `
        <div class="outer-row">
            <div class="inner-row">
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

async function UpdateDICOMInformation(dcm_idx)
{
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
        let ImgPlaceholder = document.getElementById('RawImgInner');

        if (ImgPlaceholder && ImgPlaceholder.height !== 0)
        {
            PredeterminedHeight = String(ImgPlaceholder.height) + 'px';
        }

        document.getElementById('DICOMOverview').innerHTML =
        `
            Index: ${dcm_idx_}
            </br>
            Raw File Path: ${dicom_pair_fp[0]}
            </br>
            Clean File Path: ${dicom_pair_fp[1]}
            </br>
            Patient's Original ID: ${dicom_pair['raw_dicom_metadata']['00100020'].value}
            </br>
            Total number of altered tags (excluding the pixel data): ${total_altered_dicom_tags}
        `;

        document.getElementById('ConversionResult').innerHTML = 
        `
            </br>
            <center>
                <div style="padding: 5px;">
                    Tags Display Mode
                </div>
                <input type="checkbox" id="ToggleDiff" class="toggleCheckbox" oninput="ShowDiff(this.checked)">
                <label for="ToggleDiff" class="toggleContainer" title="All: Shows all tags\nOnly Altered: Shows only altered tags" >
                    <div>All</div>
                    <div>Only Altered</div> 
                </label>
            </center>
            </br>
            <div style="background-color: rgba(50, 50, 55, 255); display: flex; border-radius: 3px; color: white;">
                <div class="image-row-left-label">
                    <b>IMAGES [DOWNSCALED]</b>
                </div>
                <div style="display: flex; flex-direction: column; flex: 1;">
                    <center>
                        <div style="font-weight: bold; padding: 5px; align-items: center;">RAW</div>
                    </center>
                    <div id="RawImg" style="flex: 1; border: 1px solid black; padding: 10px; background-color: rgba(50, 50, 55, 255); border-radius: 3px; min-height: ${PredeterminedHeight};">
                        <img id="RawImgInner" alt="Image 1" class="DCMImg">
                    </div>
                </div>
                <div class="cell-vertical-separator"></div>
                <div style="display: flex; flex-direction: column; flex: 1;">
                    <center>
                        <div style="font-weight: bold; padding: 5px; align-items: center;">DE-IDENTIFIED</div>
                    </center>
                    <div id="CleanedImg" style="flex: 1; border: 1px solid black; padding: 10px; background-color: rgba(50, 50, 55, 255); border-radius: 3px; min-height: ${PredeterminedHeight};">
                    </div>
                </div>
            </div>
            <div id="MetadataTable" class="metadata-table">
                ${dicom_metadata_table}
            </div>
        `;

        document.getElementById('RawImg').innerHTML =
        `
            <img id="RawImgInner" src="${raw_dicom_img_fp}" alt="Image 1" class="DCMImg">
        `;

        document.getElementById('CleanedImg').innerHTML =
        `
            <img id="CleanedImgInner" src="${cleaned_dicom_img_fp}" alt="Image 2" class="DCMImg">
        `;
    }
}

form.addEventListener
(
    'submit',
    function(event)
    {
        button.disabled = false;
    }
);

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

        if (dcm_files_response.ok && dcm_files.n_uploaded_files > 0)
        {
            n_uploaded_files = dcm_files.n_uploaded_files
            document.getElementById('UploadStatus').innerHTML = 
            `
                </br>\n
                Files Uploaded Successfully\n
                </br>\n
                Total uploaded files: ${n_uploaded_files}\n
                </br>\n
                Size of uploaded content: ${dcm_files.total_size} MB\n
                </br>\n
                </br>
            `
            document.getElementById('SubmitAnonymizationProcess').disabled = false
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

async function submit_dicom_processing_request()
{
    document.getElementById('SubmitAnonymizationProcess').disabled = true

    const clean_image = document.getElementById('clean-image');
    const retain_safe_private_input_checkbox = document.getElementById('retain-safe-private-input-checkbox');
    const retain_uids_input_checkbox = document.getElementById('retain-uids-input-checkbox');
    const retain_device_identity_input_checkbox = document.getElementById('retain-device-identity-input-checkbox');
    const retain_patient_characteristics_input_checkbox = document.getElementById('retain-patient-characteristics-input-checkbox');
    const date_processing_select = document.getElementById('date-processing-select');
    const retain_descriptors_input_checkbox = document.getElementById('retain-descriptors-input-checkbox');
    const patient_pseudo_id_prefix_input_text = document.getElementById('patient-pseudo-id-prefix-input-text');

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
    document.getElementById('DICOMSliderWrap').innerHTML = 
    `
        <input id="DICOMSlider" type="range" min="0" max="${n_uploaded_files-1}" value="0" step="1" class="slider" id="DICOMRange" style="width: 500px;" oninput="UpdateDICOMInformation(this.value)">
    `;

    UpdateDICOMInformation(0)
}