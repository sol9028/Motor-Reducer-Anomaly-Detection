from pathlib import Path
import sys,json
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.path.insert(0,str(Path('tmp/paper_deps').resolve()))
import pymupdf as f
src=Path(sys.argv[1]) if len(sys.argv)>1 else Path('보고서/논문_Overleaf/main.pdf')
dest=Path('tmp/paper_qa');dest.mkdir(exist_ok=True)
d=f.open(src)
report=[]
for i,p in enumerate(d):
    txt=p.get_text()
    blocks=p.get_text('blocks')
    outside=[b[:4] for b in blocks if b[0]<10 or b[1]<10 or b[2]>p.rect.width-10 or b[3]>p.rect.height-10]
    report.append(dict(page=i+1,chars=len(txt),text_top=txt[:90],outside_page=outside,tables=[s for s in txt.splitlines() if s.startswith(('표 ','그림 '))]))
    p.get_pixmap(matrix=f.Matrix(1.3,1.3)).save(str(dest/f'page_{i+1:02}.png'))
for start in range(0,len(d),4):
    sheet=f.open();p=sheet.new_page(width=1200,height=1770)
    for j in range(min(4,len(d)-start)):
        orig=d[start+j];img=orig.get_pixmap(matrix=f.Matrix(0.95,0.95))
        x=(j%2)*600;y=(j//2)*885
        p.insert_text((x+15,y+15),f'PAGE {start+j+1}',fontsize=11)
        p.insert_image(f.Rect(x+10,y+25,x+590,y+875),stream=img.tobytes('png'),keep_proportion=True)
    p.get_pixmap(matrix=f.Matrix(1.2,1.2)).save(str(dest/f'contact_{start//4+1:02}.png'))
(dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
