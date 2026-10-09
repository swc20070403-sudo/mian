import sys, zipfile, io, shutil
from PIL import Image
src = sys.argv[1]; tmp = src + '.new'
zin = zipfile.ZipFile(src); zout = zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    if it.filename.startswith('word/media/') and it.filename.endswith('.png'):
        im = Image.open(io.BytesIO(data))
        if im.mode in ('RGBA', 'LA', 'P'):
            im = im.convert('RGBA'); bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[3]); im = bg
        if im.width > 2400:
            im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, 'PNG', optimize=True, dpi=(370, 370)); old = len(data); data = buf.getvalue()
        print(it.filename, old, '->', len(data), im.size)
    zout.writestr(it, data)
zout.close(); shutil.move(tmp, src)
