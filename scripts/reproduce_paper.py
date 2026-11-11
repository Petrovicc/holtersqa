"""Verify the paper's archived inference, development log, scaler and selected epoch.

Run from a reviewed project checkout. The three input hashes below identify the
archived artifacts. No estimator is fitted and no split is generated.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

INPUT_HASHES = {
    'Models/New Top/model_bundle.joblib': '0b0adf27512aa4812d43f4fcc9214d7e8faed6a321c635a0b30a75925cda45df',
    'Models/New Top/training_results.json': '1f12b8171616b472a6f91d15c758f225d127d916abf1d8a49362fb56ebe2366a',
    'DBs/newDB.parquet': '2a87a3e6b634be3c5ef326a97ca1cbec7cebff775a537b7af7279f7575be2ce9',
    'DBs/newDB.parquet.config.json': 'd711b836d72050ddb6244f520b9dad1280c3781eaef10aa270308bebe071c2be',
}
ROLES = [('training','train_subjects'), ('validation','validation_subjects'),
         ('reported_evaluation','final_test_subjects')]
META = {'record_id','subject_id','i0','i1','label'}

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)

def score(y, p, coefficient):
    return float(np.mean(y == p) - coefficient * np.sum((y == 3) & (p == 1)) / max(1, np.sum(y == 3)))

def counts(y):
    return {str(k):int(np.sum(y == k)) for k in [1,2,3]}

def run(args):
    source, output = args.project_root, args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.code_root or source))
    from ekg_monitor.application.training_service import _resample_classes
    code = args.code_root or source
    for p,h in INPUT_HASHES.items():
        assert sha(source/p)==h, f"Input hash mismatch: {p}"
    bundle = joblib.load(source/'Models/New Top/model_bundle.joblib')
    bundle_meta = load_json(source/'Models/New Top/model_bundle.json')
    report = load_json(source/'Models/New Top/training_results.json')
    metric = bundle['metrics']
    assert metric == report['metrics'] == bundle_meta['metrics']
    cfg = bundle['training_config']['training']
    ds = bundle['training_config']['dataset']
    assert ds == load_json(source/'DBs/newDB.parquet.config.json')
    estimator = bundle['model']
    scaler = estimator.named_steps['scaler']
    model = estimator.named_steps['model']
    features = bundle['feature_columns']
    frame = pd.read_parquet(source/'DBs/newDB.parquet')
    assert len(frame)==64452 and np.array_equal(frame.index,np.arange(len(frame)))
    assert [k for k in frame.columns if k not in META] == features == report['feature_columns']
    assert np.isfinite(frame[features].to_numpy(dtype=float)).all()
    assert frame.record_id.astype(str).str.slice(0,3).equals(frame.subject_id.astype(str))
    assert set(frame.loc[frame.record_id.astype(str).isin(['100001','100002']),'subject_id'].astype(str))=={'100'}
    assert set(frame.loc[frame.subject_id.astype(str)=='100','record_id'].astype(str))=={'100001','100002'}
    cand = metric['evaluation_candidates']
    fsets = {tuple(e['final_test_subjects']) for e in cand}
    assert len(cand)==20 and len(fsets)==5
    pairs = set()
    for e in cand:
        T,V,E = (set(e[key]) for _,key in ROLES)
        assert not (T&V or T&E or V&E) and T|V|E==set(frame.subject_id.astype(str))
        pairs.add((tuple(e['validation_subjects']),tuple(e['final_test_subjects'])))
    assert len(pairs)==20
    best=max(cand,key=lambda e:e['validation_objective_score'])
    assert best['candidate']==19 and [e['candidate'] for e in cand if e['selected']]==[19]
    for role, key in ROLES:
        assert best[key]==metric[key]
    rows = []
    for e in cand:
        row = {k:v for k,v in e.items() if not k.endswith('_subjects')}
        for role,key in ROLES:
            row[f'{role}_subject_ids'] = '|'.join(e[key])
            row[f'{role}_original_windows'] = int(frame.subject_id.astype(str).isin(e[key]).sum())
        rows.append(row)
    write_csv(output/'development_candidates.csv',rows)
    allocations=[]; subjects=[]; prepared={}
    for role,key in ROLES:
        loc=frame.loc[frame.subject_id.astype(str).isin(metric[key])]
        before=loc.index.to_numpy()
        after=before
        if role!='reported_evaluation':
            after,_ = _resample_classes(before,loc.label.to_numpy(),cfg['class_equalization'],cfg['random_seed']+int(role=='validation'))
        prepared[role]=after
        for suffix,indices in [('before',before),('after',after)]:
            part=frame.loc[indices]
            obs=counts(part.label.to_numpy())
            source_key={'training':'train','validation':'validation','reported_evaluation':'test'}[role]+'_class_counts_'+suffix
            assert obs==metric[source_key]
            allocations.append(dict(partition=role,resampling=suffix,subject_ids='|'.join(metric[key]),windows=len(part),**{'class_'+k:v for k,v in obs.items()}))
            for (sid,rid),chunk in part.groupby(['subject_id','record_id']):
                subjects.append(dict(partition=role,resampling=suffix,subject_id=sid,record_id=rid,windows=len(chunk),**{'class_'+k:v for k,v in counts(chunk.label.to_numpy()).items()}))
    write_csv(output/'partition_class_counts.csv',allocations)
    write_csv(output/'subject_record_class_counts.csv',subjects)
    xtrain=frame.loc[prepared['training'],features].to_numpy(float)
    mean=np.mean(xtrain,axis=0);var=np.var(xtrain,axis=0)
    np.testing.assert_allclose(mean,scaler.mean_,rtol=1e-11,atol=1e-9)
    np.testing.assert_allclose(var,scaler.var_,rtol=1e-11,atol=1e-9)
    assert int(scaler.n_samples_seen_)==len(xtrain)==60906
    ones=np.ones(len(features))
    expected_scale=np.where(var==0.0,ones,np.sqrt(var))
    np.testing.assert_allclose(expected_scale,scaler.scale_,rtol=1e-11,atol=1e-9)
    val=frame.loc[prepared['validation']]
    pred_val=estimator.predict(val[features].to_numpy(float))
    yval=val.label.to_numpy()
    coef=cfg['dangerous_error_penalty']
    obj=score(yval,pred_val,coef)
    curve=report['accuracy_curve'];delta=cfg['early_stopping_min_delta']
    retained=float('-inf');epoch=0
    for n, value in enumerate(curve,1):
        if value>retained+delta:
            retained=value;epoch=n
    assert len(curve)==103 and epoch==3 and len(curve)-epoch==cfg['early_stopping_patience']==100
    assert obj==retained==curve[2]==metric['validation_objective_score'] and curve.count(obj)==1
    subset=frame.loc[prepared['reported_evaluation']]
    matrix=subset[features].to_numpy(float);truth=subset.label.to_numpy()
    pred=estimator.predict(matrix);proba=estimator.predict_proba(matrix)
    cm=confusion_matrix(truth,pred,labels=[1,2,3])
    assert cm.tolist()==report['confusion_matrix']==[[1480,762,6],[22,285,19],[0,3,162]]
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
    metrics={'accuracy':accuracy_score(truth,pred),'balanced_accuracy':balanced_accuracy_score(truth,pred),'f1_macro':f1_score(truth,pred,average='macro')}
    assert all(v==metric[k] for k,v in metrics.items())
    predictions=subset[['subject_id','record_id','i0','i1','label']].copy()
    predictions.insert(0,'prepared_dataset_row',subset.index)
    predictions.rename(columns={'i0':'start_sample_inclusive','i1':'end_sample_exclusive','label':'consensus_reference_class'},inplace=True)
    predictions['predicted_class']=pred
    for k in [1,2,3]: predictions[f'predicted_probability_class_{k}']=proba[:,k-1]
    predictions.to_csv(output/'predictions.csv',index=False)
    slices=[]
    for sid,ch in predictions.groupby('subject_id'):
        ref=ch.consensus_reference_class; p=ch.predicted_class
        for k in [1,2,3]:
            n=int((ref==k).sum());tp=int(((ref==k)&(p==k)).sum());npos=int((p==k).sum())
            slices.append(dict(subject_id=sid,quality_class=k,reference_support=n,predicted_positive=npos,true_positive=tp,recall=tp/n if n else None,precision=tp/npos if npos else None))
    write_csv(output/'per_subject_class_performance.csv',slices)
    write_csv(output/'features.csv',[{'position':i+1,'column':name,'scaler_mean':scaler.mean_[i],'scaler_population_variance':scaler.var_[i],'scaler_scale':scaler.scale_[i]} for i,name in enumerate(features)])
    positions=np.linspace(0,5000,33).astype(int)
    write_csv(output/'fft_bins.csv',[dict(feature=features[8+i],first_positive_rfft_index=int(lo+1),last_positive_rfft_index=int(hi),first_Hz=(lo+1)/10,last_Hz=hi/10) for i,(lo,hi) in enumerate(zip(positions[:-1],positions[1:]))])
    hashes = {p:sha(source/p) for p in list(INPUT_HASHES)+['Models/New Top/model_bundle.json','DBs/newDB.parquet.diagnostics.json','butqdb/SHA256SUMS.txt']}
    bundle_bytes=(source/'Models/New Top/model_bundle.joblib').read_bytes()
    embedded=re.search(rb'_sklearn_version\x94\x8c.([0-9]+\.[0-9]+\.[0-9]+)\x94',bundle_bytes)
    assert embedded and embedded.group(1)==b'1.8.0'
    codefiles=['ekg_monitor/application/training_service.py','ekg_monitor/application/signal_analysis.py','ekg_monitor/infrastructure/model_repository.py','ekg_monitor/infrastructure/butqdb_repository.py','ekg_monitor/application/server_service.py','ekg_monitor/application/client_service.py','ekg_monitor/infrastructure/protocol.py']
    summary={
      'audit_date':date.today().isoformat(),'dataset_source':'https://physionet.org/content/butqdb/1.0.0/',
      'input_paths_relative_to_project_root':hashes,'code_paths_relative_to_code_root':{p:sha(code/p) for p in codefiles},
      'run_identity':{'archive_directory':'Models/New Top','bundle_sha256':hashes['Models/New Top/model_bundle.joblib'],'format':bundle['artifact_format'],'created_at':bundle['created_at'],'historical_run_uuid':None,'historical_code_commit':None,'seed':cfg['random_seed'],'selected_candidate':19},
      'reproduction_runtime':{'python':platform.python_version(),'platform':platform.system(),'architecture':platform.machine(),**{n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','scikit-learn','joblib','pyarrow','wfdb','matplotlib']}},
      'historical_environment':{'backend':metric['training_backend'],'scikit_learn_serialization_version':embedded.group(1).decode(),'source_training_service_filesystem_mtime_UTC':datetime.fromtimestamp((code/codefiles[0]).stat().st_mtime,timezone.utc).isoformat(),'immutable_code_snapshot':None,'complete_package_lock':None},
      'candidate_selection':{'logged_candidates':len(cand),'max_validation_objective_candidate':best['candidate'],'max_reported_accuracy_candidate':max(cand,key=lambda e:e['final_accuracy'])['candidate'],'max_reported_balanced_accuracy_candidate':max(cand,key=lambda e:e['final_balanced_accuracy'])['candidate'],'max_reported_macro_f1_candidate':max(cand,key=lambda e:e['final_macro_f1'])['candidate'],'separate_selection_of_archived_run_over_other_runs':None},
      'scaler':{'n_samples_seen':int(scaler.n_samples_seen_),'all_prepared_features_finite':True,'mean_verified':True,'population_variance_verified':True,'scale_verified':True,'rtol':1e-11,'atol':1e-9},
      'mlp':{'parameters':model.get_params(),'hidden_activation':model.activation,'output_activation':model.out_activation_,'batch_size_effective':min(200,len(xtrain)),'completed_epochs':len(curve),'retained_epoch':epoch,'weights_shapes':[list(v.shape) for v in model.coefs_],'fitted_estimator_loss_curve_matches_log':bool(np.array_equal(model.loss_curve_,report['loss_curve'])),'post_restore_validation_class_support':counts(yval),'post_restore_validation_confusion_matrix':confusion_matrix(yval,pred_val,labels=[1,2,3]).tolist(),'post_restore_validation_objective':obj,'epoch_5_objective':curve[4],'min_delta':delta,'patience':cfg['early_stopping_patience'],'dangerous_error_coefficient':coef},
      'metrics':metrics,'classification_report':classification_report(truth,pred,labels=[1,2,3],output_dict=True),'confusion_matrix':cm.tolist(),'correct_windows':int(np.trace(cm)),'discordant_windows':int(len(pred)-np.trace(cm)),
      'commands':['python scripts/reproduce_paper.py --project-root PATH_TO_ARCHIVE --code-root . --output-dir paper_check','python scripts/verify_paper_waveforms.py --project-root PATH_TO_ARCHIVE --code-root . --predictions paper_check/predictions.csv --output-dir waveform_check'],
      'preexisting_patient_derived_data_notice':'BUT QDB v1.0.0 (PhysioNet), CC BY 4.0. Source credit and transformations are in docs/paper_reproducibility.md.'}
    write_json(output/'paper_manifest.json',summary)
    print(f"Verified {len(pred)} predictions, {len(cand)} logged candidates, selected epoch {epoch}, and {len(features)} scaler features.")

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--code-root',type=Path,help='Reviewed checkout containing ekg_monitor (defaults to project root)')
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())

