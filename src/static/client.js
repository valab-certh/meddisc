document.getElementById('myForm').addEventListener
(
    'submit',
    async function(e)
    {
        e.preventDefault();
        const formData = new FormData(this);
        const response = await fetch
        (
            '/upload_files/',
            {
                method: 'POST',
                body: formData
            }
        );
        // Response from python script server
        const result = await response.json();
        console.log(result.message);
        if (response.ok)
        {
            document.getElementById('SubmitUpload').disabled = true
            document.getElementById('UploadStatus').innerHTML = `</br>\nFiles Uploaded Successfully\n</br>\nTotal uploaded files: ${result.n_uploaded_files}\n</br>\nSize of uploaded content: ${result.total_size} MB\n</br>\n</br>`
        }
        else
        {
            document.getElementById('SubmitUpload').innerHTML = "Retry Submitting";
        }
    }
);

async function submit_dicom_processing_request()
{
    const retain_safe_private_input_checkbox = document.getElementById('retain-safe-private-input-checkbox');
    const retain_uids_input_checkbox = document.getElementById('retain-uids-input-checkbox');
    const retain_device_identity_input_checkbox = document.getElementById('retain-device-identity-input-checkbox');
    const retain_patient_characteristics_input_checkbox = document.getElementById('retain-patient-characteristics-input-checkbox');
    const date_processing_select = document.getElementById('date-processing-select');
    const retain_descriptors_input_checkbox = document.getElementById('retain-descriptors-input-checkbox');
    const patient_pseudo_id_prefix_input_text = document.getElementById('patient-pseudo-id-prefix-input-text');

    const data =
    {
        'retain_safe_private_input_checkbox': retain_safe_private_input_checkbox.checked,
        'retain_uids_input_checkbox': retain_uids_input_checkbox.checked,
        'retain_device_identity_input_checkbox': retain_device_identity_input_checkbox.checked,
        'retain_patient_characteristics_input_checkbox': retain_patient_characteristics_input_checkbox.checked,
        'date_processing_select': date_processing_select.value,
        'retain_descriptors_input_checkbox': retain_descriptors_input_checkbox.checked,
        'patient_pseudo_id_prefix_input_text': patient_pseudo_id_prefix_input_text.value
    };

    // This object is what the server receives
    const response = await fetch
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