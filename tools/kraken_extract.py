"""Pull one file out of Kraken's split OHLCVT archive with HTTP range requests,
without downloading the ~9.5 GB archive.

  python tools/kraken_extract.py datasets/XBTUSD_60.csv XBTUSD_60.csv
  python tools/kraken_extract.py datasets/ETHUSD_60.csv ETHUSD_60.csv
  python tools/kraken_extract.py datasets/ BNBUSD_60.csv ADAUSD_60.csv ...   # several into a folder

Needs assets.kraken.com reachable. Part sizes are for the archive as of 2026-10.
"""
import io, sys, zipfile, requests

BASE = "https://assets.kraken.com/marketing/institutions/Kraken_OHLCVT_Full.zip.part{:02d}"
SIZES = [2097152000] * 4 + [1157945626]


class RemoteSplitFile(io.RawIOBase):
    def __init__(self):
        self.starts, s = [], 0
        for size in SIZES:
            self.starts.append(s); s += size
        self.size, self.pos, self.fetched = s, 0, 0
        self.session = requests.Session()
        self.cache = {}  # (part, block) -> bytes
        self.block = 1 << 20

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get_block(self, part, b):
        key = (part, b)
        if key not in self.cache:
            lo = b * self.block
            hi = min(lo + self.block, SIZES[part]) - 1
            r = self.session.get(BASE.format(part), headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
            r.raise_for_status()
            assert r.status_code == 206, r.status_code
            self.cache[key] = r.content
            self.fetched += len(r.content)
        return self.cache[key]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            part = max(i for i, st in enumerate(self.starts) if st <= self.pos)
            off = self.pos - self.starts[part]
            b = off // self.block
            data = self._get_block(part, b)
            i = off - b * self.block
            chunk = data[i:i + n]
            if not chunk:
                break
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf)); buf[:len(d)] = d; return len(d)


f = RemoteSplitFile()
z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
names = z.namelist()
print(f"archive: {len(names):,} files, index read with {f.fetched/1e6:.1f} MB downloaded")
folder = sys.argv[1].endswith("/")
for target in sys.argv[2:]:
    want = [n for n in names if n.split("/")[-1].upper() == target.upper()]
    print("matches:", want, [round(z.getinfo(n).compress_size / 1e6, 1) for n in want])
    if want:
        data = z.read(want[0])
        out = sys.argv[1] + want[0].split("/")[-1] if folder else sys.argv[1]
        open(out, "wb").write(data)
        print(f"extracted {len(data)/1e6:.1f} MB -> {out}; total downloaded {f.fetched/1e6:.1f} MB of {f.size/1e9:.2f} GB")
