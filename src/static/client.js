var form = document.getElementById('UploadForm');
var button = document.getElementById('SubmitAnonymizationProcess');


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

        if (response.ok)
        {
            document.getElementById('UploadStatus').innerHTML = `</br>\nFiles Uploaded Successfully\n</br>\nTotal uploaded files: ${result.n_uploaded_files}\n</br>\nSize of uploaded content: ${result.total_size} MB\n</br>\n</br>`
            document.getElementById('SubmitAnonymizationProcess').disabled = false
        }
        else
        {
            document.getElementById('SubmitUpload').innerHTML = "Retry Submitting";
        }

    }
);

document.querySelector('#SessionForm input[name="file0"]').addEventListener('change', async function(e)
{
    e.preventDefault();
    const fileInput = this;
    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = async function() {
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
});

async function submit_dicom_processing_request()
{
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

    const files = await fetch
    (
        '/submit_button_clicked',
        {
            method: 'POST',
            headers:
            {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        }
    );
}