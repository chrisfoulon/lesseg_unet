import random
import datetime
import dateutil
from dateutil.parser import *
from pathlib import Path
import json

import pandas as pd
import numpy as np

from lesseg_unet.visualisation_utils import display_img


class TextColour:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


class Highlight:
    RED_YELLOW = '\033[2;31;43m'
    BLACK_CYAN = '\033[2;30;46m'
    RESET = '\033[0;0m'


def colour_text(text, colour):
    return colour + text + TextColour.RESET


def open_json(path):
    with open(path, 'r') as j:
        return json.load(j)


def save_json(path, d):
    with open(path, 'w+') as j:
        return json.dump(d, j, indent=4)


def format_date(date):
    if isinstance(date, str):
        return dateutil.parser.parse(date)
        # return datetime.datetime.strptime(date, '%Y%m%d')
    elif isinstance(date, datetime.datetime):
        return date
    elif isinstance(date, pd.Timestamp):
        return date.to_pydatetime()
    else:
        raise ValueError(f'Unrecognised date type: {date} of type {type(date)}')


def check_wrong_tail(seg_dict, seg_folder, wrong_tail_keys_vol_dict, new_df, nonanon_summaries, output_dict_path,
                     text_filter=''):
    pd.set_option('display.max_colwidth', None)
    columns = ['dateOfAdmissionDV', 'hospitalNumber', 'initialNIHSS', 'dateOfBirth', 'diagnosisFreetext',
               'historyFreetext', 'CTResultFreetext', 'MRIResultFreetext']
    ssnap_columns = ['ProClinV1Id',
                     'S1HospitalNumber',
                     'S1AgeOnArrival',
                     'S2BrainImagingDateTime'] + [c for c in new_df.columns if 'S2NihssArrival' in c]
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    green_text = ['left', 'Left', 'right', 'Right', 'LSW', 'RSW']
    # 'S2NihssArrivalMotorArmRight', 'S2NihssArrivalMotorLegRight', 'S2NihssArrivalBestLanguage',
    # 'S2NihssArrivalMotorArmLeft', 'S2NihssArrivalMotorLegLeft'
    # column = 'S2NihssArrivalMotorArmRight'
    # for k in wrong_tail_dict[column]:
    before_delta = datetime.timedelta(days=28)
    after_delta = datetime.timedelta(days=28)
    if Path(output_dict_path).is_file():
        new_exclude_dict = open_json(output_dict_path)
    else:
        new_exclude_dict = {}
    try:
        for num, k in enumerate(wrong_tail_keys_vol_dict):
            print(f'[{num}/{len(wrong_tail_keys_vol_dict)}]')
            if k in new_exclude_dict:
                continue
            if 'ssnap' in seg_dict[k] and seg_dict[k]['ssnap']:
                date = format_date(seg_dict[k]['StudyDate'])
                print(f'Lesion Volume: {wrong_tail_keys_vol_dict[k]}')
                print(f'############SSNAP info {k}###############')
                matched_date = False
                for ref in seg_dict[k]['ssnap']:
                    row = new_df[new_df['ProClinV1Id'] == int(ref)][ssnap_columns]
                    ssnap_date = row['S2BrainImagingDateTime'][row['S2BrainImagingDateTime'].index[0]].to_pydatetime()
                    print(date)
                    print(ssnap_date)
                    date_before = ssnap_date - before_delta
                    date_after = ssnap_date + after_delta
                    if date_before < date < date_after:
                        matched_date = True
                        print(ref)
                        ssnap_text = ''
                        for ind, col in enumerate(row):
                            val = row[col][row[col].index[0]]
                            if isinstance(val, np.float64) and val > 0:
                                val = colour_text(str(val), TextColour.GREEN)
                            else:
                                val = str(val)
                            if ind % 2:
                                ssnap_text += ' | ' + col + ': ' + val + '\n'
                            else:
                                ssnap_text += col + ': ' + val
                        print(ssnap_text)
                if not matched_date:
                    continue
                print(f'############END OF FREE TEXT###############')
                found = False
                print(f'Image PatientID {seg_dict[k]["PatientID"]}')
                print(f'############ PDF free text for session {k} ###############')
                if 'summary' in seg_dict[k] and seg_dict[k]['summary']:
                    for pdf_ref in seg_dict[k]['summary']:
                        row = nonanon_summaries[nonanon_summaries['fileName'] == pdf_ref][columns]
                        for col in row:
                            if text_filter == '':
                                found = True
                            elif row[col].str.contains(text_filter).iloc[0]:
                                found = True
                        if found:
                            for col in row:
                                if 'text' in col.lower():
                                    text = str(row[col]).replace(col, Highlight.BLACK_CYAN + col + Highlight.RESET)
                                    for t in green_text:
                                        text = text.replace(t, TextColour.GREEN + t + TextColour.RESET)
                                    print(text)
                                else:
                                    print(row[col])
                    print(f'############END OF FREE TEXT###############')
                display_img(Path(seg_folder, seg_dict[k]['b1000']),
                            Path(seg_folder, seg_dict[k]['segmentation']),
                            display='fsleyes')
                resp = input(
                    f'Keep[keep] or exclude[{set([new_exclude_dict[kk]["exclude"] for kk in new_exclude_dict])}]?\n')
                if resp.lower() in ['quit', 'q', 'exit']:
                    break
                new_exclude_dict[k] = seg_dict[k]
                new_exclude_dict[k]['exclude'] = resp
        save_json(output_dict_path, new_exclude_dict)
    except Exception as e:
        save_json(output_dict_path, new_exclude_dict)
        raise e
    return new_exclude_dict


if __name__ == '__main__':
    seg_folder = '/home/tolhsadum/remote_ucl/media/chrisfoulon/DATA1/a_imagepool_mr/' \
                 'ischemic_b1000_segmentation_UNETR_april2022/'
    seg_dict = open_json(Path(seg_folder, 'b1000_segmentation_info_dict.json'))
    non_imaging_folder = '/home/tolhsadum/remote_ucl/media/chrisfoulon/HDD2/non_imaging_data/'
    reports = pd.read_csv(Path(non_imaging_folder, 'report_df_2013_2019.csv'), header=0)
    nonanon_summaries = pd.read_excel(Path(
        '/home/tolhsadum/remote_ucl/media/chrisfoulon/DATA1/a_TXU_workspace/z1_stroke_info/b_PDF_Reports/'
        'c_reports_database2013to2019/HASUDatabase2013To2019.xls'),
                                      header=0)
    new_df = pd.read_excel(
        '/home/tolhsadum/remote_ucl/media/chrisfoulon/DATA1/a_imagepool_mr/ForChris/'
        'SSNAP_Export_281_Locked_20131101_20211031.xlsx',
        header=0)
    # Need to zfill the patientID to actually match because leading 0s were removed ...
    new_df['S1HospitalNumber'] = [str(pid).zfill(8) for pid in new_df['S1HospitalNumber']]
    wrong_tail_keys_vol_dict = open_json('/home/tolhsadum/remote_ucl/media/chrisfoulon/HDD2/data_linking_tables/'
                                         'ssnap_mismatch_check/ranked_wrong_tail_keys_to_volume.json')
    exclude_dict_path = '/home/tolhsadum/remote_ucl/media/chrisfoulon/HDD2/data_linking_tables/' \
                        'ssnap_mismatch_check/exclude_mismatch_keys_code.json'
    exclude_dict = check_wrong_tail(seg_dict, seg_folder, wrong_tail_keys_vol_dict, new_df,
                                    nonanon_summaries, exclude_dict_path)
