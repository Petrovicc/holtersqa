"""Reproduce reported predictions from the released bundle and original BUT QDB files."""
import argparse
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from reproduce_paper import INPUT_HASHES, load_json, sha, write_json

def main(args):
    sys.path.insert(0,str(args.code_root))
    from ekg_monitor.application.models import DatasetBuildConfig, FeatureConfig, FilterConfig
    from ekg_monitor.application.training_service import build_dataset
    from ekg_monitor.infrastructure.butqdb_repository import ButqdbRepository

    bundle_path=args.bundle_dir/'model_bundle.joblib'
    report_path=args.bundle_dir/'training_results.json'
    assert sha(bundle_path)==INPUT_HASHES['Models/New Top/model_bundle.joblib']
    assert sha(report_path)==INPUT_HASHES['Models/New Top/training_results.json']
    bundle=joblib.load(bundle_path)
    report=load_json(report_path)
    config=dict(bundle['training_config']['dataset'])
    config['feature_config']=FeatureConfig(**config['feature_config'])
    config['filter_config']=FilterConfig(**config['filter_config'])
    repo=ButqdbRepository(args.dataset_root)
    subjects=set(report['metrics']['final_test_subjects'])
    records=sorted(r.record_id for r in repo.records.values() if r.subject_id in subjects)
    assert records==['104001','114001','115001','123001','125001','126001']
    checked={}
    sums={k:h for h,k in (line.split(maxsplit=1) for line in (args.dataset_root/'SHA256SUMS.txt').read_text().splitlines() if line.strip())}
    for record in records:
        for suffix in ['ECG.hea','ECG.dat','ACC.hea','ACC.dat','ANN.csv']:
            rel=f'{record}/{record}_{suffix}'
            digest=sha(args.dataset_root/rel)
            assert digest==sums[rel],f'Dataset hash mismatch: {rel}'
            checked[rel]=digest
    frame=build_dataset(repo,records,DatasetBuildConfig(**config))
    assert len(frame)==2739
    cols=bundle['feature_columns']
    assert [c for c in frame.columns if c not in ['subject_id','record_id','i0','i1','label']]==cols
    x=frame[cols].to_numpy(dtype=float)
    assert np.isfinite(x).all()
    ref=frame.label.to_numpy(int)
    model=bundle['model'];pred=model.predict(x);prob=model.predict_proba(x)
    cm=confusion_matrix(ref,pred,labels=[1,2,3])
    assert cm.tolist()==report['confusion_matrix']==[[1480,762,6],[22,285,19],[0,3,162]]
    m={'accuracy':accuracy_score(ref,pred),'balanced_accuracy':balanced_accuracy_score(ref,pred),'f1_macro':f1_score(ref,pred,average='macro')}
    assert all(v==report['metrics'][k] for k,v in m.items())
    keys=['subject_id','record_id','start_sample_inclusive','end_sample_exclusive','consensus_reference_class']
    found=frame[['subject_id','record_id','i0','i1','label']].copy()
    found.columns=keys
    found['predicted_class']=pred
    pcols=[f'predicted_probability_class_{k}' for k in (1,2,3)]
    found[pcols]=prob
    expected=pd.read_csv(args.reference_predictions,dtype={'record_id':str,'subject_id':str},float_precision='round_trip')
    for name in ['record_id','subject_id']:
        found[name]=found[name].astype(str)
    joined=found.merge(expected.drop(columns=['prepared_dataset_row']),on=keys,how='outer',indicator=True,validate='one_to_one',suffixes=('_raw','_stored'))
    assert len(joined)==2739 and (joined._merge=='both').all()
    assert (joined.predicted_class_raw==joined.predicted_class_stored).all()
    p0=joined[[k+'_raw' for k in pcols]].to_numpy();p1=joined[[k+'_stored' for k in pcols]].to_numpy()
    np.testing.assert_allclose(p0,p1,atol=1e-12,rtol=1e-12)
    out=args.output_dir;out.mkdir(exist_ok=True,parents=True)
    found.to_csv(out/'predictions_from_raw.csv',index=False)
    write_json(out/'public_reproduction.json',{'verification_date':date.today().isoformat(),'source':'https://physionet.org/content/butqdb/1.0.0/','source_raw_paths_and_SHA256':checked,'bundle_sha256':sha(bundle_path),'reference_predictions_sha256':sha(args.reference_predictions),'records':records,'subjects':sorted(subjects),'rows_rebuilt_from_consensus_and_raw_samples':len(frame),'same_identifiers_and_reference_labels':True,'same_2739_class_predictions':True,'probability_maximum_absolute_difference':float(np.abs(p0-p1).max()),'probability_tolerance':{'rtol':1e-12,'atol':1e-12},'confusion_matrix':cm.tolist(),'metrics':m,'historical_training_reproduced':False})
    print(f'Rebuilt {len(frame)} evaluation windows directly from BUT QDB. All class predictions matched. Maximum probability difference: {np.abs(p0-p1).max():.3g}.')

if __name__=='__main__':
    checkout=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True,help='Local BUT QDB 1.0.0 root with SHA256SUMS.txt and the six indicated record directories')
    parser.add_argument('--bundle-dir',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,default=checkout)
    parser.add_argument('--reference-predictions',type=Path,default=checkout/'docs/reproducibility/predictions.csv')
    parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args())

