"""Trace the five S1 examples to WFDB and compare one stored window with the server.

The verification runs the production packet path with real signal blocks. Its
socket pair carries usual server ACKs. The test does not measure transport time.
"""
import argparse
import hashlib
import json
import socket
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import wfdb
from reproduce_paper import INPUT_HASHES, sha, write_json

def arraysig(array):
    return hashlib.sha256(np.ascontiguousarray(array,dtype='<f8').tobytes()).hexdigest()

def main(args):
    archive=args.project_root
    code=args.code_root or archive
    out=args.output_dir
    out.mkdir(exist_ok=True,parents=True)
    sys.path.insert(0,str(code))
    from ekg_monitor.application.server_service import ServerService,ClientState
    from ekg_monitor.application.signal_analysis import apply_filter, extract_feature_map
    from ekg_monitor.infrastructure import model_repository
    from ekg_monitor.infrastructure.butqdb_repository import ButqdbRepository,interval_label
    from ekg_monitor.infrastructure.signal_history import DiskSignalHistory
    from ekg_monitor.infrastructure.protocol import SignalBlock,encode_signal_blocks
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for key in ['Models/New Top/model_bundle.joblib','DBs/newDB.parquet']:
        assert sha(archive/key)==INPUT_HASHES[key]
    source=ButqdbRepository(archive/'butqdb')
    official={k:v for v,k in (line.strip().split(maxsplit=1) for line in (archive/'butqdb/SHA256SUMS.txt').read_text().splitlines() if line.strip())}
    model_path=archive/'Models/New Top/model_bundle.joblib'
    loaded=model_repository.load_model(model_path)
    data=pd.read_parquet(archive/'DBs/newDB.parquet')
    pred=pd.read_csv(args.predictions)
    errors=pred.loc[pred['consensus_reference_class']!=pred['predicted_class']]
    assert len(pred)==2739 and len(errors)==812
    keys=['consensus_reference_class','predicted_class']
    examples=errors.sort_values(keys+['record_id','start_sample_inclusive']).groupby(keys,sort=True).head(1).copy()
    assert len(examples)==5
    evidence=[];waveforms=[];rawhashes={};parity=None
    fig,axes=plt.subplots(5,1,figsize=(11,12),constrained_layout=True)
    for n,((_,row),ax) in enumerate(zip(examples.iterrows(),axes),1):
        rid=str(int(row['record_id']));begin=int(row['start_sample_inclusive']);end=int(row['end_sample_exclusive'])
        sid=str(int(row['subject_id']));reference=int(row[keys[0]]);quality=int(row[keys[1]])
        record=source.get(rid);eid='E'+str(n)
        assert record.subject_id==sid and interval_label(source.annotation_groups(rid),begin,end,'consensus')==reference
        for suffix in ['ANN.csv','ECG.hea','ECG.dat','ACC.hea','ACC.dat']:
            rel=f'{rid}/{rid}_{suffix}'
            if rel not in rawhashes:
                digest=sha(archive/'butqdb'/rel)
                assert digest==official[rel],f'BUT QDB SHA256SUMS mismatch: {rel}'
                rawhashes[rel]=digest
        raw_record=wfdb.rdrecord(str(archive/'butqdb'/rid/(rid+'_ECG')),sampfrom=begin,sampto=end,channels=[0],physical=True)
        ecg=np.asarray(raw_record.p_signal[:,0],float)
        assert len(ecg)==end-begin==10000 and raw_record.fs==1000 and raw_record.units[0]=='uV'
        duration=len(ecg)/raw_record.fs
        samples=source.read_ecg_sample_window(rid,begin,end)
        e,a=samples['ecg'],samples['acc']
        np.testing.assert_array_equal(e['samples'][:,0],ecg.astype(np.float32))
        filtered=apply_filter(e['samples'],e['fs'],loaded.filter_config)
        mapping=extract_feature_map(filtered,a['samples'],record.metadata,loaded.feature_config,a['fs'])
        assert list(mapping)==loaded.feature_columns
        vec=np.array([list(mapping.values())])
        idx=int(row['prepared_dataset_row']);stored=data.iloc[idx]
        assert [str(stored.record_id),int(stored.i0),int(stored.i1),int(stored.label)]==[rid,begin,end,reference]
        supplied=stored[loaded.feature_columns].to_numpy(float).reshape(1,-1)
        actual_class=int(loaded.estimator.predict(supplied)[0]);assert actual_class==quality
        actual_proba=loaded.estimator.predict_proba(supplied)[0]
        reproduce_class=int(loaded.estimator.predict(vec)[0])
        feature_difference=float(np.max(np.abs(supplied-vec)))
        diff_proba=float(np.max(np.abs(loaded.estimator.predict_proba(vec)[0]-actual_proba)))
        times=np.arange(len(ecg))/raw_record.fs
        start_seconds=begin/raw_record.fs;end_seconds=end/raw_record.fs
        waveforms.append(pd.DataFrame(dict(example_id=eid,record_id=rid,ecg_sample=np.arange(begin,end),time_in_window_seconds=times,raw_ecg_uV=ecg)))
        title=f'{eid}  Record {rid}  |  {start_seconds:.3f}-{end_seconds:.3f} s  |  consensus {reference} → predicted {quality}'
        def draw(ax):
            ax.plot(times,ecg,linewidth=.65,color='#17456A')
            ax.set_title(title,loc='left',fontsize=10)
            ax.set_xlabel('Time within window (s)');ax.set_ylabel(f'Raw ECG ({raw_record.units[0]})')
            ax.grid(alpha=.18);ax.set_xlim(0,10)
        draw(ax)
        one,axs=plt.subplots(figsize=(10,2.8),constrained_layout=True);draw(axs)
        one.savefig(out/f'{eid}_raw_ECG.png',dpi=200)
        plt.close(one)
        evidence.append(dict(example_id=eid,prepared_dataset_row=idx,record_id=rid,subject_id=sid,start_sample_inclusive=begin,end_sample_exclusive=end,ecg_sampling_hz=e['fs'],acc_sampling_hz=a['fs'],ecg_units=e['units'][0],acc_units='|'.join(a['units']),samples=len(ecg),duration_seconds=duration,consensus_class_rechecked=reference,archived_feature_prediction=actual_class,directly_extracted_window_prediction=reproduce_class,max_abs_raw_to_prepared_feature_difference=feature_difference,max_abs_probability_difference=diff_proba,physical_ECG_float64_little_endian_sha256=arraysig(ecg)))
        if n==1:
            # Identical sample positions and block boundaries on both paths. Record
            # actual input to the production estimator without replacing its output.
            ss=ServerService();online=ss.import_model(model_path)
            seen=[];inner=model_repository.features_for_columns
            def capture_vector(*args,**kwargs):
                result=inner(*args,**kwargs);seen.append(result.copy());return result
            sent_classes=[];history=DiskSignalHistory('paper-check')
            left,right=socket.socketpair()
            try:
                state=ClientState(rid,('127.0.0.1',0),left,'paper-check',{},2.,history)
                with patch.object(model_repository,'features_for_columns',side_effect=capture_vector):
                    for j in range(5):
                        blocks=[]
                        for name,block in samples.items():
                            step=round(2*block['fs'])
                            blocks.append(SignalBlock(name,block['samples'][j*step:(j+1)*step],block['fs'],block['channel_names'],block['units']))
                        descriptor,payload=encode_signal_blocks(blocks)
                        ss._process_data(state,{'subject_id':rid,'sequence':j+1,'end_sample':begin+round((j+1)*2*e['fs']),'streams':descriptor},payload)
                        sent_classes.append(int(state.last_assessment.quality_class))
                assert len(seen)==1 and sent_classes==[0,0,0,0,quality]
                np.testing.assert_allclose(seen[0],vec,rtol=1e-12,atol=1e-9)
                np.testing.assert_allclose(seen[0],supplied,rtol=1e-12,atol=1e-9)
                zs=online.estimator.named_steps['scaler'].transform(seen[0])
                zo=loaded.estimator.named_steps['scaler'].transform(supplied)
                np.testing.assert_allclose(zs,zo,rtol=1e-12,atol=1e-9)
                server_probs=online.estimator.predict_proba(seen[0])[0]
                np.testing.assert_allclose(server_probs,actual_proba,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(state.last_assessment.quality_score,max(actual_proba),rtol=1e-12,atol=1e-12)
                parity={'test_date':date.today().isoformat(),'classification':'new deterministic functional check','example_id':eid,'record_id':rid,'start_ecg_sample':begin,'end_ecg_sample_exclusive':end,'packet_seconds':2,'samples_per_packet':{'ecg':int(2*e['fs']),'acc_per_axis':int(2*a['fs'])},'class_after_each_of_five_packets':sent_classes,'source_and_server_feature_order_equal':True,'rtol_features_scaled_values_probabilities':1e-12,'atol_features_and_scaled_values':1e-9,'atol_probabilities':1e-12,'raw_recomputed_vs_prepared_features_max_abs':feature_difference,'server_vs_prepared_features_max_abs':float(np.max(np.abs(seen[0]-supplied))),'server_vs_offline_features_max_abs':float(np.max(np.abs(seen[0]-vec))),'server_vs_prepared_scaled_values_max_abs':float(np.max(np.abs(zs-zo))),'server_vs_prepared_probabilities_max_abs':float(np.max(np.abs(server_probs-actual_proba))),'offline_probabilities':actual_proba.tolist(),'server_probabilities':server_probs.tolist(),'server_confidence':state.last_assessment.quality_score,'expected_class':quality,'same_class':True}
            finally:
                left.close();right.close();history.close()
    fig.suptitle('Illustrative discordant windows from the reconstructed MLP predictions\nEarliest window for each observed error direction; labels have not been re-adjudicated',fontsize=12)
    fig.savefig(out/'MLP_illustrative_error_examples.png',dpi=200);fig.savefig(out/'MLP_illustrative_error_examples.pdf')
    plt.close(fig)
    pd.concat(waveforms).to_csv(out/'raw_ECG_examples.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(evidence).to_csv(out/'waveform_checks.csv',index=False)
    write_json(out/'waveform_provenance.json',{'date':date.today().isoformat(),'source':'https://physionet.org/content/butqdb/1.0.0/','dataset_version':'1.0.0','license':'CC BY 4.0','path_base':'BUT QDB v1.0.0 root','canonical_file_SHA256_verified':rawhashes,'annotation_policy':'Consensus intervals in source ANN file, verified for the recorded [start, end) ECG sample positions.','plot_inputs':'WFDB physical ECG in µV; first time sample is zero within each excerpt. Stored float64 array digests are over row-major little-endian physical amplitudes.','selection':'One example for each observed error direction. Sort by reference class, predicted class, record ID and start sample, then retain the first of each direction.','expert_readjudication':False})
    write_json(out/'server_parity.json',parity)
    print(f'Verified five 10-second excerpts and one five-packet server replay. Server class: {parity["expected_class"]}.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--code-root',type=Path)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args())

