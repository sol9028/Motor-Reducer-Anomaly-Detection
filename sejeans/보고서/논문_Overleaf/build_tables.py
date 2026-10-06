"""Rebuild manuscript tables from the project's saved evaluation JSON (stdlib only)."""
from pathlib import Path
import json, csv, statistics, hashlib

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
TABLES = OUT / 'tables'
TABLES.mkdir(exist_ok=True)
B = ROOT / '본격/모델_12ch_베이스라인'
M = ROOT / '본격/모델/모델'
SOURCES = {
    '12ch': B/'모델_12ch_베이스라인/fusion_eval.json',
    '15ch': M/'eval_results_fuse_rpm.json',
    '18ch': M/'eval_results_fuse_rpm_temp.json',
    '48ch': M/'eval_results_fuse_rpm_ext_no_band_ratio_7.json',
    '51ch': B/'모델_ord_A/eval_results_fuse_rpm_ext.json',
    '51ord': B/'모델_ord_only/eval_results_ord_only.json',
    '81ch': B/'모델_ord_C/eval_results_fuse_rpm_ext_no_band_ratio_7_no_ord_band_ratio_7.json',
    '87ch': B/'모델_ord_B/eval_results_fuse_rpm_ext.json',
    'k1250': B/'모델_커널스윕/eval_results_fuse_rpm_ext_ord_k1250.json',
    'k2500': B/'모델_커널스윕/eval_results_fuse_rpm_ext_ord_k2500.json',
    'k5000': B/'모델_커널스윕/eval_results_fuse_rpm_ext_ord_k5000.json',
    '15gk': M/'eval_results_fuse_rpm_gk.json',
}
DATA = {k:json.loads(p.read_text(encoding='utf-8')) for k,p in SOURCES.items()}
LAT_PATHS = {k: ROOT/'엣지벤치'/f'{k}.json' for k in ['result_t1','result_t4','result_t4_b8','result_t1_b32']}
LAT = {k:json.loads(p.read_text(encoding='utf-8')) for k,p in LAT_PATHS.items()}

def recall(r, cls='DEMAG'):
    i=r['labels'].index(cls)
    return r['confusion_matrix'][i][i]/sum(r['confusion_matrix'][i])

def errors(r, a='DEMAG', b='NORMAL'):
    return r['confusion_matrix'][r['labels'].index(a)][r['labels'].index(b)]

def mean_f1(key): return statistics.mean(r['macro_f1_mean'] for r in DATA[key])
def mean_rec(key): return statistics.mean(recall(r) for r in DATA[key])
def write(name, s): (TABLES/name).write_text(s,encoding='utf-8')
def tabular(spec,head,rows):
    return '\\begin{tabular}{'+spec+'}\n\\toprule\n'+head+' \\\\\n\\midrule\n'+'\n'.join(r+' \\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'

for key,rows in DATA.items():
    for r in rows:
        assert sum(map(sum,r['confusion_matrix']))==r['n_chunks'],(key,r['slot'])
        assert r['n_splits']==3

audit=[]
for key,rows in DATA.items():
    for r in rows:
        audit.append(dict(experiment=key,vehicle=r['slot'][0],chunks=r['n_chunks'],sessions=r.get('n_sessions',''),
                          macro_f1_mean=r['macro_f1_mean'],macro_f1_std=r['macro_f1_std'],accuracy_mean=r['acc_mean'],
                          demag_recall=recall(r),demag_to_normal=errors(r),normal_to_demag=errors(r,'NORMAL','DEMAG'),
                          source=str(SOURCES[key].relative_to(ROOT))))
with (OUT/'verified_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)

desc={'12ch':'기본 특징','15ch':'기본 + rpm','18ch':'기본 + rpm + 온도','48ch':'절대축 11개 추가','51ch':'절대축 12개 추가','51ord':'차수축 12개 추가','81ch':'절대축 11 + 차수축 11','87ch':'절대축 12 + 차수축 12'}
rows=[]
for key in ['12ch','15ch','18ch','48ch','51ch','51ord','81ch','87ch']:
    n='51 (차수축)' if key=='51ord' else key[:-2]
    rows.append(n+' & '+desc[key]+' & '+' & '.join(f"{r['macro_f1_mean']:.4f}" for r in DATA[key])+f' & {mean_f1(key):.4f}')
write('ablation.tex',tabular('llrrrr','채널 & 특징 구성 & IONIQ & KONA & NIRO & 평균',rows))

rows=[]
for r in DATA['87ch']:
    counts=r['n_sessions_per_class']
    rows.append(r['slot'][0]+f" & {r['n_chunks']:,} & {r['n_sessions']} & "+' & '.join(str(counts[c]) for c in r['labels']))
write('samples.tex',tabular('lrrrrrrr','차종 & 청크 & 주행 & DEMAG & ECC10 & ECC20 & NORMAL & REDUC',rows))

rows=[]
for r in DATA['87ch']:
    rows.append(r['slot'][0]+f" & ${r['macro_f1_mean']:.4f}\\pm{r['macro_f1_std']:.4f}$ & ${r['acc_mean']:.4f}\\pm{r['acc_std']:.4f}$ & {recall(r):.4f}")
write('final.tex',tabular('lrrr','차종 & Macro-F1 & Accuracy & DEMAG 재현율',rows))

rows=[]
for a,b in zip(DATA['51ch'],DATA['87ch']):
    rows.append(a['slot'][0]+f' & {recall(a):.4f} & {recall(b):.4f} & {(recall(b)-recall(a))*100:.2f} & {errors(a)} & {errors(b)} & {errors(a,"NORMAL","DEMAG")} & {errors(b,"NORMAL","DEMAG")}')
write('demag.tex',tabular('lrrrrrrr','차종 & $R_{51}$ & $R_{87}$ & $\\Delta R$ (pp) & $D\\to N_{51}$ & $D\\to N_{87}$ & $N\\to D_{51}$ & $N\\to D_{87}$',rows))

K=[('k1250',1176,0.10),('k2500',2436,0.20),('k5000',4956,0.36),('87ch',9996,0.69)]
rows=[]
for i,(key,k,size) in enumerate(K):
    lr=LAT['result_t1']['rows'][i]
    rows.append(f'{k:,} & {mean_f1(key):.4f} & {mean_rec(key):.4f} & {size:.2f} & {lr["total_ms_p50"]:.2f} & {lr["total_ms_p95"]:.2f} & {lr["total_ms_p99"]:.2f}')
write('kernels.tex',tabular('rrrrrrr','변환 특징 수 & Macro-F1 & DEMAG 재현율 & 모델 MB & p50 ms & p95 ms & p99 ms',rows))

rows=[]
for key,k,size in K:
    i=K.index((key,k,size))
    rows.append(f'{k:,} & '+' & '.join(f'{LAT[cond]["rows"][i]["per_chunk_ms_p50"]:.2f}' for cond in LAT))
write('latency.tex',tabular('rrrrr','변환 특징 수 & $t1,b1$ & $t4,b1$ & $t4,b8$ & $t1,b32$',rows))

for r in DATA['87ch']:
    rows=[label+' & '+' & '.join(str(v) for v in row) for label,row in zip(r['labels'],r['confusion_matrix'])]
    write('cm_'+r['slot'][0].lower()+'.tex',tabular('lrrrrr','실제 / 예측 & DEMAG & ECC10 & ECC20 & NORMAL & REDUC',rows))

curve='features f1 recall latency\n'+'\n'.join(f'{k} {mean_f1(key):.8f} {mean_rec(key):.8f} {LAT["result_t1"]["rows"][i]["total_ms_p50"]}' for i,(key,k,size) in enumerate(K))
write('kernel_curve.dat',curve+'\n')

dn_a=sum(errors(r) for r in DATA['51ch']);dn_b=sum(errors(r) for r in DATA['87ch'])
bi_a=sum(errors(r)+errors(r,'NORMAL','DEMAG') for r in DATA['51ch'])
bi_b=sum(errors(r)+errors(r,'NORMAL','DEMAG') for r in DATA['87ch'])
summary=dict(final_macro_f1=mean_f1('87ch'),final_demag_recall=mean_rec('87ch'),
             d_to_n_51=dn_a,d_to_n_87=dn_b,d_to_n_reduction=(dn_a-dn_b)/dn_a,
             bidirectional_51=bi_a,bidirectional_87=bi_b,bidirectional_reduction=(bi_a-bi_b)/bi_a,
             total_chunks=sum(r['n_chunks'] for r in DATA['87ch']),total_sessions=sum(r['n_sessions'] for r in DATA['87ch']))
(OUT/'verified_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in list(SOURCES.values())+list(LAT_PATHS.values())]
(OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
