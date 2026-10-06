from pathlib import Path
import sys, zipfile, xml.etree.ElementTree as ET, zlib, struct
sys.path.insert(0, str(Path('tmp/paper_deps').resolve()))
import fitz, olefile
out=Path('tmp/paper_sources'); out.mkdir(parents=True,exist_ok=True)
files=list(Path('기존연구').glob('*.pdf'))+list(Path('발표자료').glob('*.pdf'))+list(Path('보고서').glob('*.pdf'))+list(Path('보고서/방학지료').glob('*.pdf'))
for p in files:
    d=fitz.open(p)
    text='\n'.join(f'\n--- PAGE {i+1} ---\n'+pg.get_text() for i,pg in enumerate(d))
    (out/(p.stem+'.txt')).write_text(text,encoding='utf-8')
    print(p,len(d),len(text))
for p in Path('보고서').glob('*.docx'):
    with zipfile.ZipFile(p) as z:
        root=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text='\n'.join(''.join(n.itertext()) for n in root.findall('.//w:t',ns))
        (out/(p.stem+'_docx.txt')).write_text(text,encoding='utf-8')
for p in Path('보고서').glob('*데이터설명서*.hwp'):
    ole=olefile.OleFileIO(p)
    head=ole.openstream('FileHeader').read()
    compressed=bool(struct.unpack_from('<I',head,36)[0]&1)
    texts=[]
    for stream in ole.listdir():
        if stream[0]=='BodyText' and stream[-1].startswith('Section'):
            b=ole.openstream(stream).read()
            if compressed:b=zlib.decompress(b,-15)
            at=0
            while at+4<=len(b):
                h=struct.unpack_from('<I',b,at)[0];at+=4
                tag=h&1023;size=h>>20
                if size==4095:size=struct.unpack_from('<I',b,at)[0];at+=4
                payload=b[at:at+size];at+=size
                if tag==67:texts.append(payload.decode('utf-16le',errors='replace'))
    (out/(p.stem+'.txt')).write_text('\n'.join(texts),encoding='utf-8')
    print(p,len(texts))
