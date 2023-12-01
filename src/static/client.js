var form = document.getElementById('UploadForm');
var button = document.getElementById('SubmitAnonymizationProcess');
var slider = document.getElementById("DICOMRange");
var n_uploaded_files;
var dicom_pair_fps;


async function updateValue(dcm_idx)
{
    function createTable(metadata)
    {
        let tableHTML =
        `
            <table>\n
                <tr>\n
                    <th>Tag ID</th>\n
                    <th>VR</th>\n
                    <th>Tag Name</th>\n
                    <th>Tag Value</th>\n
                </tr>\n
        `;

        for (let key in metadata)
        {
            let tagID = key;
            let vr = metadata[key].vr;
            let tagName = metadata[key].name;
            let tagValue = metadata[key].value;
    
            tableHTML +=
            `
                <tr>
                    <td>${tagID}</td>
                    <td>${vr}</td>
                    <td>${tagName}</td>
                    <td>${tagValue}</td>
                </tr>
            `;
        }

        tableHTML += '</table>';

        return tableHTML;
    }

    const dicom_pair_fp = dicom_pair_fps[dcm_idx]
    const response = await fetch
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

    if (response.ok)
    {
        const dicom_pair = await response.json();
        const raw_dicom_metadata_table = createTable(dicom_pair['raw_dicom_metadata']);
        const cleaned_dicom_metadata_table = createTable(dicom_pair['cleaned_dicom_metadata']);
        const raw_dicom_img_fp = dicom_pair['raw_dicom_img_fp']
        const cleaned_dicom_img_fp = dicom_pair['cleaned_dicom_img_fp']

        document.getElementById('DICOMIDX').innerHTML = 
        `
            Index: ${dcm_idx}\n
            </br>\n
            Raw File Path: ${dicom_pair_fp[0]}\n
            </br>\n
            Clean File Path: ${dicom_pair_fp[1]}\n
        `;

        document.getElementById('ConversionResult').innerHTML = 
        `
            <center>
                <div style="font-weight: bold;">Raw</div>
            </center>
            <center>
                <div style="font-weight: bold;">Clean</div>
            </center>
            <div style="border: 1px solid black; padding: 10px;">
                <img src="${raw_dicom_img_fp}" alt="Image 1" style="width: 100%; height: auto; object-fit: cover;">
            </div>
            <div style="border: 1px solid black; padding: 10px;">
                <img src="${cleaned_dicom_img_fp}" alt="Image 2" style="width: 100%; height: auto; object-fit: cover;">
            </div>
            <div style="border: 1px solid black; padding: 10px;">
                ${raw_dicom_metadata_table}
            </div>
            <div style="border: 1px solid black; padding: 10px;">
                ${cleaned_dicom_metadata_table}
            </div>
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

        const response = await fetch
        (
            '/upload_files/',
            {
                method: 'POST',
                body: formData
            }
        );
        const result = await response.json();

        if (response.ok && result.n_uploaded_files > 0)
        {
            n_uploaded_files = result.n_uploaded_files
            document.getElementById('UploadStatus').innerHTML = 
            `
                </br>\n
                Files Uploaded Successfully\n
                </br>\n
                Total uploaded files: ${n_uploaded_files}\n
                </br>\n
                Size of uploaded content: ${result.total_size} MB\n
                </br>\n
                </br>
            `
            document.getElementById('SubmitAnonymizationProcess').disabled = false
        }
        else
        {
            alert('Invalid directory input. Make sure it contains at least one DICOM file.')
        }
    }
);

document.querySelector('#SessionForm input[name="file0"]').addEventListener
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

    const response = await fetch
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

    dicom_pair_fps = await response.json()

    document.getElementById('DICOMBarWrap').innerHTML = 
    `
        <input id="DICOMBar" type="range" min="0" max="${n_uploaded_files-1}" value="0" step="1" class="slider" id="DICOMRange" style="width: 500px;" oninput="updateValue(this.value)">
    `

    updateValue(0)
}