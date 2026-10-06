"""Create a single-file manuscript and a minimal Overleaf upload archive."""
from pathlib import Path
import re, zipfile
OUT=Path(__file__).resolve().parent
source=(OUT/'main.tex').read_text(encoding='utf-8')
source=source.replace('\\newcommand{\\smalltable}[1]{\\begingroup\\small\\setlength{\\tabcolsep}{5pt}\\input{tables/#1}\\endgroup}\n','')
def inline_table(m):
    return '\\begingroup\\small\\setlength{\\tabcolsep}{5pt}\n'+(OUT/'tables'/m[1]).read_text(encoding='utf-8')+'\\endgroup'
source=re.sub(r'\\smalltable\{([^}]+)\}',inline_table,source)
data=(OUT/'tables/kernel_curve.dat').read_text(encoding='utf-8').strip()
source=source.replace('table[x=latency,y=f1,meta=features]{tables/kernel_curve.dat}',
                      'table[x=latency,y=f1,meta=features,row sep=\\\\]{\n'+data.replace('\n',' \\\\\n')+' \\\\\n}')
source=source.replace('% Numeric tables are generated from saved JSON by build_tables.py.',
                      '% Standalone version: all tables and chart data are embedded.\n% Generated from the modular main.tex by package_overleaf.py.')
assert '\\smalltable{' not in source
assert '\\input{tables/' not in source
assert '{tables/kernel_curve.dat}' not in source
(OUT/'paper_single.tex').write_bytes(source.encode('utf-8'))
dest=OUT.parent/'논문_Overleaf.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('main.tex',source)
    for name in ['README.md','근거_검증_및_확인사항.md','verified_metrics.csv','verified_summary.json','source_manifest.json','validation.json']:
        z.write(OUT/name,name)
print(dest)
