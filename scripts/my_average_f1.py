import json
from collections import defaultdict
import argparse
import numpy as np
import os

LANG2VEC_LANGS = "am,bn,cs,de,el,es,et,eu,fi,fr,hi,hu,hy,id,ja,jv,ko,la,lv,mi,my,pt,ru,tr,vi,zh"
LANG2VEC_LANGS_EN = "en,am,bn,cs,de,el,es,et,eu,fi,fr,hi,hu,hy,id,ja,jv,ko,la,lv,mi,my,pt,ru,tr,vi,zh"
ALL_LANGS= "am,ar,bxr,cdo,cs,de,el,en,es,et,eu,fa,fi,fr,gn,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,kv,la,lv,mhr,mi,my,myv,pt,qu,ru,se,sw,tk,tr,vi,wo,xmf,zh,zh_yue"
URIEL_LANGS = "am,bn,cs,de,el,en,es,et,eu,fi,fr,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,la,lv,mhr,mi,my,myv,pt,ru,se,tk,tr,vi,wo"
INFILE = 'outputs/{}/my-bert-base-multilingual-cased-MaxLen128_{}_{}/{}_results.txt'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='en')
    parser.add_argument('--languages', default='en')
    parser.add_argument('--task', default='ner')
    parser.add_argument('--split', default='dev')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--all_lang', dest='all_lang', action='store_true')
    parser.add_argument('--lang2vec_lang', dest='lang2vec_lang', action='store_true')
    parser.add_argument('--lang2vec_lang_en', dest='lang2vec_lang_en', action='store_true')
    parser.add_argument('--uriel_lang', dest='uriel_lang', action='store_true')
    parser.add_argument('--en_weight', default=0.1)
    parser.set_defaults(all_lang=False, lang2vec_lang=False)
    args = parser.parse_args()
    if args.all_lang:
        args.languages = ALL_LANGS
    if args.lang2vec_lang:
        args.languages = LANG2VEC_LANGS
    if args.uriel_lang:
        args.languages = URIEL_LANGS
    if args.lang2vec_lang_en:
        args.languages = LANG2VEC_LANGS_EN

    if args.task == 'ner':
        dataset = 'panx'
    else:
        dataset = 'udpos'
    
    if args.method == 'en':
        infile = INFILE.format(dataset, args.task, 'en', args.split)
    elif args.method == 'related':
        infile = INFILE.format(dataset, args.task, args.languages, args.split)
    elif args.method == 'ensemble':
        infile = INFILE.format(dataset, args.task, f'{args.languages}_ensemble_attribution', args.split)
    elif args.method == 'ensemble_en':
        infile = INFILE.format(dataset, args.task, f'{args.languages}_ensemble_awequal_en_en{args.en_weight}', args.split)
    elif args.method == 'emeas1':
        infile = INFILE.format(dataset, args.task, f'{args.languages}_emea_s1', args.split)
    elif args.method == 'emeas10':
        infile = INFILE.format(dataset, args.task, f'{args.languages}_emea_s10', args.split)
    elif args.method == 'syntax_ensemble':
        infile = INFILE.format(dataset, args.task, f'{args.languages}_syntax_ensemble_en{args.en_weight}', args.split)
    elif args.method == 'learned_ensemble':
        infile = INFILE.format(dataset, args.task, f'{args.languages}_learned_ensemble_temp{args.temperature}', args.split)
    else:
        print('INVALID METHOD')
        exit()
    print(infile)
    
    results = defaultdict(dict)
    average_results = defaultdict(float)
    std_results = defaultdict(float)
    with open(infile, 'r') as f:
        for line in f:
            line = json.loads(line)
            language = line['language']
            seed = line['seed']
            if seed not in [1,2,3]: continue
            f1 = line['f1']
            results[language][seed] = f1

    for language in results.keys():
        results[language] = [100*results[language][seed] for seed in results[language].keys()]
        num_seeds = len(results[language])
        # average_results[language] = average_results[language]/num_seeds
        # average_results[language] = np.mean([results[language][seed] for seed in results[language].keys()])
        # std_results[language] = np.std([results[language][seed] for seed in results[language].keys()])
        average_results[language] = np.mean(results[language])
        std_results[language] = np.std(results[language])
        # print(results[language])
        print(f'Average f1 score over {num_seeds} seeds for language {language} = {format(average_results[language],".2f")} +- {format(std_results[language],".2f")}')
if __name__ == '__main__':
    main()

    