import json
import re

import nlp2

# ls -l | grep '^d' | grep ' ks' | awk '{print $9}'
# ls -l | grep '^d' | grep ' asr' | awk '{print $9}'
# ls -l | grep '^d' | grep ' esc50' | awk '{print $9}'

results_folder = [
    './ks_hubert_layer6_code100',
    './ks_hubert_layer6_code200',
    './ks_hubert_layer6_code50',
    './ks_hubert_layer9_code500',
    './ks_mhubert_layer11_code1000',
    './esc50_hubert_layer6_code100_fold_1',
    './esc50_hubert_layer6_code100_fold_2',
    './esc50_hubert_layer6_code100_fold_3',
    './esc50_hubert_layer6_code100_fold_4',
    './esc50_hubert_layer6_code100_fold_5',
    './esc50_hubert_layer6_code200_fold_1',
    './esc50_hubert_layer6_code200_fold_2',
    './esc50_hubert_layer6_code200_fold_3',
    './esc50_hubert_layer6_code200_fold_4',
    './esc50_hubert_layer6_code200_fold_5',
    './esc50_hubert_layer6_code50_fold_1',
    './esc50_hubert_layer6_code50_fold_2',
    './esc50_hubert_layer6_code50_fold_3',
    './esc50_hubert_layer6_code50_fold_4',
    './esc50_hubert_layer6_code50_fold_5',
    './esc50_hubert_layer9_code500_fold_1',
    './esc50_hubert_layer9_code500_fold_2',
    './esc50_hubert_layer9_code500_fold_3',
    './esc50_hubert_layer9_code500_fold_4',
    './esc50_hubert_layer9_code500_fold_5',
    './esc50_mhubert_layer11_code1000_fold_1',
    './esc50_mhubert_layer11_code1000_fold_2',
    './esc50_mhubert_layer11_code1000_fold_3',
    './esc50_mhubert_layer11_code1000_fold_4',
    './esc50_mhubert_layer11_code1000_fold_5',
    './esc50_hubert_layer9_code500_fold_1_norm',
    './esc50_hubert_layer9_code500_fold_2_norm',
    './esc50_hubert_layer9_code500_fold_3_norm',
    './esc50_hubert_layer9_code500_fold_4_norm',
    './esc50_hubert_layer9_code500_fold_5_norm',
    './esc50_hubert_layer9_code500_fold_1_beam',
    './esc50_hubert_layer9_code500_fold_2_beam',
    './esc50_hubert_layer9_code500_fold_3_beam',
    './esc50_hubert_layer9_code500_fold_4_beam',
    './esc50_hubert_layer9_code500_fold_5_beam',
    './esc50_hubert_layer9_code500_fold_1_norm_beam',
    './esc50_hubert_layer9_code500_fold_2_norm_beam',
    './esc50_hubert_layer9_code500_fold_3_norm_beam',
    './esc50_hubert_layer9_code500_fold_4_norm_beam',
    './esc50_hubert_layer9_code500_fold_5_norm_beam'
]

for result_folder in results_folder:
    rcsv = [['filename', 'epoch', 'score']]
    for f in nlp2.get_files_from_dir(result_folder, "_greedy_each_data_score.csv"):
        result_csv = nlp2.read_csv(f)
        match = re.search(r'(?<=/)\d+(?=\.)', f)
        if match:
            file_number = match.group()
            accuracy = 0
            total = 0
            for i in result_csv:
                if i[0] == i[1]:
                    accuracy += 1
                total += 1
            rcsv.append([f, file_number, accuracy / total])
    nlp2.write_csv(rcsv, result_folder.replace(".", "").replace("/", "") + '_result.csv')

results_folder = [
    './asr_hubert_layer6_code100',
    './asr_hubert_layer6_code200',
    './asr_hubert_layer6_code50',
    './asr_hubert_layer9_code500',
    './asr_mhubert_layer11_code1000',
    './asr_hubert_layer9_code500_norm_True_beam_False',
    './asr_hubert_layer9_code500_norm_False_beam_True',
    './asr_hubert_layer9_code500_norm_True_beam_True',
]
for result_folder in results_folder:
    rcsv = [['filename', 'epoch', 'score']]
    for f in nlp2.get_files_from_dir(result_folder, "_greedy_filtersim_False_score.csv"):
        result_csv = nlp2.read_csv(f)
        match = re.search(r'(?<=/)\d+(?=\.)', f)
        if match:
            file_number = match.group()
            rcsv.append([f, file_number, json.loads(",".join(result_csv[1]).replace("\'", "\""))['WER']])
    nlp2.write_csv(rcsv, result_folder.replace(".", "").replace("/", "") + '_result.csv')
