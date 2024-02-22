from urllib import request
import pandas as pd

def main() -> None:
	acceptable_actions = {'', 'K', 'C', 'Z', 'X'}
	with request.urlopen('https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_e.html') as response:
		html = response.read()
	df_tables = pd.read_html(html)
	df_table = df_tables[3]
	df_table = df_table.fillna('')
	dicom_tags = df_table['Tag'].str.replace('[(,)]', '', regex=True).to_list()
	basic_profile = ['X' for i in df_table['Basic Prof.'].to_list()]
	retain_safe_private_option = df_table['Rtn. Safe Priv. Opt.'].to_list()
	retain_uids_option = df_table['Rtn. UIDs Opt.'].to_list()
	retain_device_identity_option = df_table['Rtn. Dev. Id. Opt.'].to_list()
	retain_patient_characteristics_option = ['K' if i == 'C' else i for i in df_table["Rtn. Pat. Chars. Opt."].to_list()]
	retain_long_modified_dates_option = df_table['Rtn. Long. Modif. Dates Opt.']
	retain_description_option = \
	[
		"K" if i == "C" else i for i in df_table["Clean Desc. Opt."].to_list()
	]
	output = \
	[
		basic_profile,
		retain_safe_private_option,
		retain_uids_option,
		retain_device_identity_option,
		retain_patient_characteristics_option,
		retain_long_modified_dates_option,
		retain_description_option,
	]
	output = list(map(list, zip(*output)))
	dicom_tag_to_nema_action = dict(zip(dicom_tags, output))
	dicom_tag_to_nema_action_df = pd.DataFrame(dicom_tag_to_nema_action).transpose()
	dicom_tag_to_nema_action_df.columns = ['Default', 'Rtn. Safe Priv. Opt.', 'Rtn. UIDs Opt.', 'Rtn. Dev. Id. Opt.', 'Rtn. Pat. Chars. Opt.', 'Rtn. Long. Modif. Dates Opt.', 'Clean Desc. Opt.']
	dicom_tag_to_nema_action_df.insert(0, 'Name', ['' for i in range(len(dicom_tag_to_nema_action_df))])
	dicom_tag_to_nema_action_df.loc['00100010', 'Default'] = 'Z'
	dicom_tag_to_nema_action_df.loc['00100020', 'Default'] = 'Z'
	dicom_tag_to_nema_action_df.rename(columns = {'Clean Desc. Opt.': 'Rtn. Desc. Opt.'}, inplace = True)
	dicom_tag_to_nema_action_df.insert(loc = 6, column = 'Offset Long. Modif. Dates Opt.', value = dicom_tag_to_nema_action_df['Rtn. Long. Modif. Dates Opt.'])
	dicom_tag_to_nema_action_df.insert(loc = 7, column = 'Remove Long. Modif. Dates Opt.', value = dicom_tag_to_nema_action_df['Rtn. Long. Modif. Dates Opt.'])
	dicom_tag_to_nema_action_df.replace(to_replace = {'Rtn. Long. Modif. Dates Opt.': 'C'}, value = 'K', inplace = True)
	dicom_tag_to_nema_action_df.replace(to_replace = {'Remove Long. Modif. Dates Opt.': 'C'}, value = 'X', inplace = True)
	dicom_tag_to_nema_action_df.replace\
	(
		to_replace = \
		{
			'Rtn. Safe Priv. Opt.': 'C',
			'Rtn. UIDs Opt.': 'C',
			'Rtn. Dev. Id. Opt.': 'C',
			'Rtn. Pat. Chars. Opt.': 'C',
			'Rtn. Desc. Opt.': 'C'
		},
		value = 'K',
		inplace = True
	)
	for tag_idx in dicom_tag_to_nema_action_df.index:
		tag = '(' + tag_idx[0:4] + ',' + tag_idx[4:9] + ')'
		if len(tag_idx) > 8:
			assert tag_idx == 'ggggeeee where gggg is odd', 'E: Tag index error'
			tag = '(' + tag_idx[0:4] + ',' + tag_idx[4:8] + ')' + tag_idx[8:]
		dicom_tag_to_nema_action_df.loc[tag_idx, 'Name'] = df_table.loc[df_table['Tag'] == tag, 'Attribute Name'].item()
		if tag_idx in ['00100010', '00100020']:
			dicom_tag_to_nema_action_df.loc[tag_idx, 'Default'] = 'Z'
		elif "K" not in dicom_tag_to_nema_action_df.loc[tag_idx].values.tolist()[1:]:
			dicom_tag_to_nema_action_df.loc[tag_idx, 'Default'] = 'X'
	dicom_tag_to_nema_action_df.sort_values(by = 'Name', inplace = True)
	assert all(dicom_tag_to_nema_action_df.loc['00100010'].to_numpy()[2:] == '') and all(dicom_tag_to_nema_action_df.loc['00100020'].to_numpy()[2:] == ''), 'E: Patient ID (00100010) and patient name (00100020) actions must remain empty for every action group except from the default one'
	dicom_tag_to_nema_action_arr = dicom_tag_to_nema_action_df.iloc[:, 1:].to_numpy()
	set_ = set()
	for row in dicom_tag_to_nema_action_arr:
		for value in row:
			set_.add(value)
	assert set_ <= acceptable_actions, 'E: Unacceptable values found in action table'
	dicom_tag_to_nema_action_df.to_csv('./tmp/action-groups-dcm.csv')

if __name__ == "__main__":
	main()