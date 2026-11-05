"""Select genuine DeepScaleR questions without changing ground-truth rewards."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata


def key(text):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(text)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path(__file__).parent/'data/deepscaler_train.jsonl')
    ap.add_argument('--count',type=int,default=32)
    ap.add_argument('--seed',type=int,default=36)
    ap.add_argument('--eval-source',type=Path,required=True)
    ap.add_argument('--minimum-question-characters',type=int,default=180)
    ap.add_argument('--maximum-question-characters',type=int,default=1400)
    args=ap.parse_args()
    rows=json.loads(args.source.read_text());seen=set();selected=[];eval_questions=set();filtered_overlap=0
    if args.eval_source.exists():
        import pyarrow.parquet as pq
        for row in pq.read_table(args.eval_source).to_pylist():
            question=row.get('problem') or row.get('question') or row.get('Problem')
            if question:eval_questions.add(key(question))
    for index,row in enumerate(rows):
        question=str(row.get('problem') or row.get('question') or '')
        answer=row.get('answer',row.get('ground_truth'))
        if answer is None or not args.minimum_question_characters<=len(question)<=args.maximum_question_characters:continue
        normalized=key(question)
        if normalized in eval_questions:
            filtered_overlap+=1
            continue
        if normalized in seen:continue
        seen.add(normalized)
        selected.append({'question':question+'\n\nUse the Python tool at least once to check a calculation before giving the final boxed answer.',
                         'ground_truth':str(answer),'data_source':'deepscaler_math',
                         'source_row_index':index,'original_question':question})
    random.Random(args.seed).shuffle(selected)
    selected=selected[:args.count]
    if len(selected)!=args.count:raise RuntimeError('Insufficient genuine source questions')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected))
    manifest={'source':str(args.source),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
              'dataset':'agentica-org/DeepScaleR-Preview-Dataset','seed':args.seed,'count':len(selected),
              'filtered_eval_overlap':filtered_overlap,'eval_source':str(args.eval_source),
              'eval_source_sha256':hashlib.sha256(args.eval_source.read_bytes()).hexdigest() if args.eval_source.exists() else None,
              'selected_source_indices':[r['source_row_index'] for r in selected],
              'reward':'original mathematical answer correctness; no fabricated reward or tool-call reward',
              'output_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
