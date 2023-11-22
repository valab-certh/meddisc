# DICOM De-Identifier

The executable provided in this repository serves as a tool for the removal of patient's personal identifiable information from DICOM file metadata, based on a user's options. This implementation aligns with the standards specified in Nema's Attribute Confidentiality Profiles that can be found at [[link](https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_e.html)].

## Utilities

A file named as `.json` contains all the currently available user options, within the following simple JSON configuration structure. One such file may contain
```
{
    "input_dcm_fp": "../in.dcm",
    "output_dcm_fp": "../out.dcm",
    "clean_image": true,
    "retain_safe_private": false,
    "retain_uids": false,
    "retain_device_identity": false,
    "retain_patient_characteristics": false,
    "date_processing": "remove",
    "retain_descriptors": false,
    "patient_pseudo_id_prefix": "<PREFIX ID> - "
}
```
This particular configuration is defined to be the default one, in which case the DICOM file simply applies the `Default` column from `action_groups_dcm.csv`. This last file is a basis for all possible de-identification operations, where each column corresponds to one user option. A user can define their configuration through `user_input.json`.

To explain each field

- `input_dcm_fp`. Type string. specifies the input DICOM file's path.
- `output_dcm_fp`. Type string. specifies the path of the exported DICOM file.

From now and on each explained field corresponds to a parameter in the de-identification methodology, 

- `clean_image`. Type boolean. If set to `true` then the image data is processed by the image de-identifier, effectively removing any potential burned-in text to the image pixel data that may or may not contain patient PII.
- `retain_safe_private`. Type boolean. If set to true, then for all non-empty actions of `Rtn. Safe Priv. Opt.`, the de-identification algorithm overrides the corresponding actions from the `Default` column.
- `retain_uids`. Type boolean. If set to true, it overrides the `Rtn. UIDs Opt.` column.
- `retain_device_identity`. Type boolean. If set to true, it overrides the `Rtn. Dev. Id. Opt.` column.
- `retain_patient_characteristics`. Type boolean. If set to true, it overrides the `Rtn. Pat. Chars. Opt.` column.
- `date_processing`. Type string. Value set `{offset, remove, keep}`. If option is set to
    - `offset`, it overrides the `Offset Long. Modif. Dates Opt.` column.
    - `remove`, it overrides the `Remove Long. Modif. Dates Opt.` column.
    - `remove`, it overrides the `Rtn. Long. Modif. Dates Opt.` column.
- `retain_descriptors`. Type boolean. If set to true, it overrides the `Rtn. Desc. Opt.` column.
- `patient_pseudo_id_prefix`. Type string. The concatenation of that prefix with a dummy number results in the pseudo patient ID.

## Technical Description

### Metadata Nema Actions

Below is a list of the currently implemented **actions** for the de-identifier.

#### Remove Tag - Code `X`

Replaces tag value with empty string. E.g.
```
dcm[<TAG_INDEX>].value = ''
```

#### Clean Tag - Code `C`

Implemented only for VRs that are either `DA` or `TM`.

`DA` -> Such tags are date tags in the format `YYYYMMDD`. The way that `C` is applied in such tags is by taking `YYYYMMDD` and adding to that date a random offset number of $\texttt{days}$ which is sampled from the following uniform distribution $\mathfrak{U}$, as follows
$$
\texttt{days} \sim \mathfrak{U}(365 \cdot 10, 2 \cdot (365 \cdot 10))
$$

`TM` -> Day time tags in the format `HHMMSS.FFFFFF`. The way that `C` is applied in such a tag is by simply replacing its value with a random offset number of $\texttt{seconds}$ (hence there is no second-fraction), sampled as
$$
\texttt{seconds} \sim \mathfrak{U}(0, 3600 \cdot 24)
$$

#### Keep Tag - `K`

Simply keeps a tag as is.

#### Replace Tag - `Z`

Replaces tag value with a dummy one. Implemented only for patient ID with tag index `(0010, 0020)` and patient Name with tag index `(0010, 0010)` in which case both values are replaced by a common pseudo patient ID.

### Directory Structure

```
.
├── .gitignore
├── action_groups_dcm.csv
├── in.dcm
├── README.md
├── requested_action_group_dcm.csv
├── requirements.txt
├── src
│   ├── action_tools.py
│   ├── generate_action_groups.py
│   ├── main.py
│   ├── rw.py
│   ├── script.js
│   └── server.py
├── user_default_input.json
└── user_input.json
```

## References

- Paschalis Bizopoulos: [dicom-de-identification-and-curation-tool](https://github.com/pbizopoulos/dicom-de-identification-and-curation-tool)