import tempfile
from datasets import load_dataset
def T(name, fn):
    try:
        r = fn()
        print(f"[{name}] OK -> {r!r}"[:900])
    except Exception as e:
        print(f"[{name}] EXC {type(e).__name__}: {e}"[:900])
td = tempfile.mkdtemp()
def w(name, text, enc):
    p = f"{td}/{name}"
    with open(p, "w", encoding=enc, newline="") as f:
        f.write(text)
    return p
FA = ">s1 first\nACGT\n>s2 second\nGG\n"
FQ = "@r1 first\nACGT\n+\nIIII\n@r2\nGG\n+\n@@\n"
GB = """LOCUS       TEST1                      8 bp    DNA     linear   UNK 01-JAN-1980
DEFINITION  first record.
ACCESSION   TEST1
VERSION     TEST1.1
FEATURES             Location/Qualifiers
ORIGIN
        1 acgtacgt
//
LOCUS       TEST2                      4 bp    DNA     linear   UNK 01-JAN-1980
DEFINITION  second record.
ACCESSION   TEST2
VERSION     TEST2.1
FEATURES             Location/Qualifiers
ORIGIN
        1 gggg
//
"""
for fmt, text, ext, cols in [("fasta", FA, "fasta", ["id", "description", "sequence"]), ("fastq", FQ, "fastq", ["id", "sequence", "quality"]), ("genbank", GB, "gb", None)]:
    for enc in ["utf-8", "utf-8-sig"]:
        p = w(f"x_{enc}.{ext}", text, enc)
        kw = {"columns": cols} if cols else {}
        def run(streaming=False):
            ds = load_dataset(fmt, data_files=p, split="train", streaming=streaming, **kw)
            rows = list(ds) if streaming else ds.to_list()
            if fmt == "genbank":
                rows = [{k: r.get(k) for k in ("locus_name", "length", "definition", "accession", "sequence")} | {"record_is_none": r.get("record") is None} for r in rows]
            return rows
        T(f"{fmt}_{enc}", run)
        T(f"{fmt}_{enc}_streaming", lambda: run(True))
