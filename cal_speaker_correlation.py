import json
from collections import defaultdict
from itertools import combinations

import editdistance
import nlp2
from numpy import dot
from scipy import stats
from tqdm.auto import tqdm

result_word = defaultdict(lambda: defaultdict(list))

word_dict = defaultdict(list)

with open('./superb_ks_validation_chunk_30_speaker.jsonl', 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    file_path = result['file'].split('/')
    word = file_path[-2]
    speaker_id = file_path[-1].split('_')[0]
    word_dict[word].append(result)

with open('./superb_ks_test_chunk_30_speaker.jsonl', 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    file_path = result['file'].split('/')
    word = file_path[-2]
    speaker_id = file_path[-1].split('_')[0]
    word_dict[word].append(result)


def cosine_similarity(vector1, vector2):
    if vector1 == 0 or vector2 == 0:
        return 0
    return dot(vector1, vector2) / ((dot(vector1, vector1) ** .5) * (dot(vector2, vector2) ** .5))


for word in word_dict.keys():
    for i in tqdm(combinations(word_dict[word], 2)):
        sample_0 = i[0]
        sample_1 = i[1]

        hubert_code_sample_0 = sample_0['hubert_layer6_code50_merged_code']
        hubert_code_sample_1 = sample_1['hubert_layer6_code50_merged_code']
        ed_hubert_code = editdistance.eval(hubert_code_sample_0, hubert_code_sample_1)
        result_word[word]['hubert_layer6_code50'].append(ed_hubert_code)

        hubert_code_sample_0 = sample_0['hubert_layer6_code100_merged_code']
        hubert_code_sample_1 = sample_1['hubert_layer6_code100_merged_code']
        ed_hubert_code = editdistance.eval(hubert_code_sample_0, hubert_code_sample_1)
        result_word[word]['hubert_layer6_code100'].append(ed_hubert_code)

        hubert_code_sample_0 = sample_0['hubert_layer6_code200_merged_code']
        hubert_code_sample_1 = sample_1['hubert_layer6_code200_merged_code']
        ed_hubert_code = editdistance.eval(hubert_code_sample_0, hubert_code_sample_1)
        result_word[word]['hubert_layer6_code200'].append(ed_hubert_code)

        hubert_code_sample_0 = sample_0['hubert_layer9_code500_merged_code']
        hubert_code_sample_1 = sample_1['hubert_layer9_code500_merged_code']
        ed_hubert_code = editdistance.eval(hubert_code_sample_0, hubert_code_sample_1)
        result_word[word]['hubert_layer9_code500'].append(ed_hubert_code)

        hubert_code_sample_0 = sample_0['mhubert_layer11_code1000_merged_code']
        hubert_code_sample_1 = sample_1['mhubert_layer11_code1000_merged_code']
        ed_hubert_code = editdistance.eval(hubert_code_sample_0, hubert_code_sample_1)
        result_word[word]['mhubert_layer11_code1000'].append(ed_hubert_code)

        x_vector_sample_0 = sample_0['x_vector']
        x_vector_sample_1 = sample_1['x_vector']
        cos_sim = cosine_similarity(x_vector_sample_0, x_vector_sample_1)
        result_word[word]['x_vector'].append(cos_sim)

        d_vector_sample_0 = sample_0['d_vector'] if len(sample_0['d_vector']) > 0 else [0] * 256
        d_vector_sample_1 = sample_1['d_vector'] if len(sample_1['d_vector']) > 0 else [0] * 256
        if len(d_vector_sample_0) != len(d_vector_sample_1):
            print(d_vector_sample_0)
            print(d_vector_sample_1)
        cos_sim = cosine_similarity(d_vector_sample_0, d_vector_sample_1)
        result_word[word]['d_vector'].append(cos_sim)

result_list = []
for word, rdict in result_word.items():
    print(word)
    try:
        print('hubert_layer6_code50 x_vector', stats.pearsonr(rdict['hubert_layer6_code50'], rdict['x_vector']))
        print('hubert_layer6_code100 x_vector', stats.pearsonr(rdict['hubert_layer6_code100'], rdict['x_vector']))
        print('hubert_layer6_code200 x_vector', stats.pearsonr(rdict['hubert_layer6_code200'], rdict['x_vector']))
        print('hubert_layer9_code500 x_vector', stats.pearsonr(rdict['hubert_layer9_code500'], rdict['x_vector']))
        print('mhubert_layer11_code1000 x_vector', stats.pearsonr(rdict['mhubert_layer11_code1000'], rdict['x_vector']))

        print('compare d_vector x_vector', stats.pearsonr(rdict['d_vector'], rdict['x_vector']))

        print('hubert_layer6_code50 d_vector', stats.pearsonr(rdict['hubert_layer6_code50'], rdict['d_vector']))
        print('hubert_layer6_code100 d_vector', stats.pearsonr(rdict['hubert_layer6_code100'], rdict['d_vector']))
        print('hubert_layer6_code200 d_vector', stats.pearsonr(rdict['hubert_layer6_code200'], rdict['d_vector']))
        print('hubert_layer9_code500 d_vector', stats.pearsonr(rdict['hubert_layer9_code500'], rdict['d_vector']))
        print('mhubert_layer11_code1000 d_vector', stats.pearsonr(rdict['mhubert_layer11_code1000'], rdict['d_vector']))

        results = [word]

        r, p = stats.pearsonr(rdict['hubert_layer6_code50'], rdict['x_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['hubert_layer6_code100'], rdict['x_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['hubert_layer6_code200'], rdict['x_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['hubert_layer9_code500'], rdict['x_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['mhubert_layer11_code1000'], rdict['x_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['d_vector'], rdict['x_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['hubert_layer6_code50'], rdict['d_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['hubert_layer6_code100'], rdict['d_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['hubert_layer6_code200'], rdict['d_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['hubert_layer9_code500'], rdict['d_vector'])
        results.extend([r, p])

        r, p = stats.pearsonr(rdict['mhubert_layer11_code1000'], rdict['d_vector'])
        results.extend([r, p])

        result_list.append(results)
    except:
        pass

nlp2.write_csv(result_list, 'speaker_correlation.csv')
