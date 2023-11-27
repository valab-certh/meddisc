var form = document.getElementById('UploadForm');
var button = document.getElementById('SubmitAnonymizationProcess');


document.getElementById('UploadForm').addEventListener
(
    // Inside this area, the expression ".getElementById(<html_element_id>)" works strictly inside the form corresponding to UploadForm.
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
        // Response from python script server (result holds whatever the server's "return" function sends)
        const result = await response.json();

        console.log(result.message);
        if (response.ok)
        {
            // document.getElementById('SubmitUpload').disabled = true
            document.getElementById('UploadStatus').innerHTML = `</br>\nFiles Uploaded Successfully\n</br>\nTotal uploaded files: ${result.n_uploaded_files}\n</br>\nSize of uploaded content: ${result.total_size} MB\n</br>\n</br>`
        }
        else
        {
            document.getElementById('SubmitUpload').innerHTML = "Retry Submitting";
        }
    }
);

document.getElementById('SessionForm').addEventListener
(
    'submit',
    async function(e)
    {
        e.preventDefault();
        const data = new FormData(this);

        // console.log(JSON.stringify(data));

        await fetch
        (
            '/session/',
            {
                method: 'POST',
                body: data
            }
        );
    }
);

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

    // This object is what the server receives
    // Downloads file
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

    // The downloaded files are then seen by the browser as a set of downloadables
    if (files.ok) {
        const blob = await files.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'de-identified-files.zip';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    }
}

// Add an event listener for the submit event on the form
form.addEventListener
(
    'submit',
    function(event)
    {
        // Enable the button
        button.disabled = false;
    }
);