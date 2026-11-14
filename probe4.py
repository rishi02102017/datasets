import tempfile, os, io, copy, json
import pyarrow as pa
from datasets import *
def T(name, fn):
    try:
        r = fn()
        print(f"[{name}] OK -> {r!r}"[:700])
    except Exception as e:
        print(f"[{name}] EXC {type(e).__name__}: {e}"[:700])
td = tempfile.mkdtemp()
# Features dict ops
def ior():
    f = Features({"a": Value("int64")}); f |= {"b": ClassLabel(names=["x","y"])}
    return (dict(f._column_requires_decoding), f.decode_example({"a": 1, "b": 0}))
T("feat_ior", ior)
T("feat_or_type", lambda: type(Features({"a": Value("int64")}) | {"b": Value("int8")}))
T("feat_ror_type", lambda: type({"b": Value("int8")} | Features({"a": Value("int64")})))
def ior_img():
    import PIL.Image
    b = io.BytesIO(); PIL.Image.new("RGB",(2,2)).save(b, format="PNG")
    f = Features({"a": Value("int64")}); f |= {"img": Image()}
    return f.decode_example({"a": 1, "img": {"bytes": b.getvalue(), "path": None}})
T("feat_ior_img", ior_img)
def ior_cast():
    import PIL.Image
    b = io.BytesIO(); PIL.Image.new("RGB",(2,2)).save(b, format="PNG")
    ds = Dataset.from_dict({"a":[1], "img":[{"bytes": b.getvalue(), "path": None}]})
    f = Features({"a": Value("int64")}); f |= {"img": Image()}
    return type(ds.cast(f)[0]["img"]), type(ds.to_iterable_dataset().cast(f).__iter__().__next__()["img"])
T("feat_ior_cast", ior_cast)
def setitem_oldlist():
    f = Features({"a": Value("int64")}); f["b"] = [Value("int32")]
    return f, f.arrow_schema
T("feat_setitem_oldlist", setitem_oldlist)
T("feat_update_oldlist", lambda: (lambda f: (f.update({"b": [Value("int32")]}), f["b"])[1])(Features({"a": Value("int64")})))
T("feat_deepcopy", lambda: copy.deepcopy(Features({"a": Image()}))._column_requires_decoding)
T("feat_fromkeys", lambda: Features.fromkeys(["a"], Value("int8"))._column_requires_decoding)
# Json feature
jf = Features({"j": Json()})
data = {"j": [{"a": 1}, [1, 2], "s", None, 3.5, {"n": {"m": [None]}}]}
T("json_from_dict", lambda: Dataset.from_dict(data, features=jf)["j"])
T("json_from_list", lambda: Dataset.from_list([{"j": v} for v in data["j"]], features=jf)["j"])
T("json_infer", lambda: Dataset.from_list([{"j": {"a": 1}}, {"j": {"a": "x"}}]).features)
T("json_to_parquet_rt", lambda: (Dataset.from_dict(data, features=jf).to_parquet(td+"/j.pq"), (lambda r: (r.features, r["j"]))(Dataset.from_parquet(td+"/j.pq")))[1])
T("json_to_json_rt", lambda: (Dataset.from_dict(data, features=jf).to_json(td+"/j.jsonl"), open(td+"/j.jsonl").read())[1])
T("json_to_csv", lambda: (Dataset.from_dict(data, features=jf).to_csv(td+"/j.csv", index=False), open(td+"/j.csv").read())[1])
T("json_to_pandas", lambda: Dataset.from_dict(data, features=jf).to_pandas()["j"].tolist())
T("json_to_polars", lambda: Dataset.from_dict(data, features=jf).to_polars()["j"].to_list())
T("json_to_list", lambda: Dataset.from_dict(data, features=jf).to_list())
T("json_np_fmt", lambda: Dataset.from_dict(data, features=jf).with_format("numpy")[0:2])
T("json_save_load_disk", lambda: (Dataset.from_dict(data, features=jf).save_to_disk(td+"/jd"), load_from_disk(td+"/jd")["j"])[1])
T("json_iterable", lambda: list(Dataset.from_dict(data, features=jf).to_iterable_dataset()))
T("json_map", lambda: Dataset.from_dict(data, features=jf).map(lambda x: x)["j"])
T("json_map_batched", lambda: Dataset.from_dict(data, features=jf).map(lambda x: x, batched=True)["j"])
T("json_filter", lambda: Dataset.from_dict(data, features=jf).filter(lambda x: x["j"] is not None)["j"])
T("json_sort_shuffle_flatten_idx", lambda: Dataset.from_dict(data, features=jf).shuffle(seed=0).flatten_indices()["j"])
T("json_concat", lambda: concatenate_datasets([Dataset.from_dict(data, features=jf)]*2).num_rows)
T("json_cast_from_string", lambda: Dataset.from_dict({"j": ['{"a":1}', '[1]']}).cast_column("j", Json())["j"])
T("json_cast_from_struct", lambda: Dataset.from_dict({"j": [{"a":1}, {"a":2}]}).cast_column("j", Json())["j"])
T("json_cast_to_string", lambda: Dataset.from_dict(data, features=jf).cast_column("j", Value("string"))["j"])
T("json_nested_list", lambda: Dataset.from_dict({"l": [[{"a":1}, "x"], []]}, features=Features({"l": List(Json())}))["l"])
T("json_nested_struct", lambda: Dataset.from_dict({"s": [{"k": {"a":1}}, {"k": None}, None]}, features=Features({"s": {"k": Json()}}))["s"])
T("json_yaml_rt", lambda: Features._from_yaml_list(Features({"j": Json(), "l": List(Json())})._to_yaml_list()))
T("json_dict_rt", lambda: Features.from_dict(Features({"j": Json()}).to_dict()))
T("json_arrow_rt", lambda: Features.from_arrow_schema(Features({"j": Json()}).arrow_schema))
T("json_unique", lambda: Dataset.from_dict(data, features=jf).unique("j"))
T("json_from_pandas", lambda: Dataset.from_pandas(__import__("pandas").DataFrame({"j": [{"a":1}, [1]]}), features=jf)["j"])
T("json_add_item", lambda: Dataset.from_dict(data, features=jf).add_item({"j": {"z": [1]}})["j"][-1])
T("json_add_column", lambda: Dataset.from_dict({"x":[1,2]}).add_column("j", [{"a":1}, [2]], feature=Json())["j"])
T("json_load_jsonl", lambda: (open(td+"/m.jsonl","w").write('{"x": {"a": 1}}\n{"x": [1,2]}\n{"x": "s"}\n'), load_dataset("json", data_files=td+"/m.jsonl", split="train").features)[1])
T("json_iter_batch", lambda: list(Dataset.from_dict(data, features=jf).to_iterable_dataset().batch(2))[0])
T("json_dd_class", lambda: Json())
# polars
import polars as pl, datetime as dt
T("pl_struct", lambda: Dataset.from_polars(pl.DataFrame({"s": [{"a": 1, "b": "x"}]})).features)
T("pl_datetime", lambda: Dataset.from_polars(pl.DataFrame({"d": [dt.datetime(2020,1,1)]})).features)
T("pl_date_tz", lambda: Dataset.from_polars(pl.DataFrame({"d": [dt.datetime(2020,1,1)]}).with_columns(pl.col("d").dt.replace_time_zone("UTC"))).features)
T("pl_duration", lambda: Dataset.from_polars(pl.DataFrame({"d": [dt.timedelta(1)]})).features)
T("pl_enum", lambda: Dataset.from_polars(pl.DataFrame({"e": ["a","b"]}, schema={"e": pl.Enum(["a","b"])})).features)
T("pl_array", lambda: Dataset.from_polars(pl.DataFrame({"a": [[1,2],[3,4]]}, schema={"a": pl.Array(pl.Int32, 2)})).features)
T("pl_null", lambda: Dataset.from_polars(pl.DataFrame({"n": [None, None]})).features)
T("pl_decimal", lambda: Dataset.from_polars(pl.DataFrame({"d": [1.5]}, schema={"d": pl.Decimal(5,2)})).features)
T("pl_binary", lambda: Dataset.from_polars(pl.DataFrame({"b": [b"x"]})).features)
T("pl_time", lambda: Dataset.from_polars(pl.DataFrame({"t": [dt.time(1,2)]})).features)
T("pl_to_polars_array2d", lambda: Dataset.from_dict({"x":[[[1,2],[3,4]]]}, features=Features({"x": Array2D((2,2),"int32")})).to_polars().schema)
T("pl_to_polars_classlabel", lambda: Dataset.from_dict({"l":[0,1]}, features=Features({"l": ClassLabel(names=["a","b"])})).to_polars().schema)
T("pl_rt_large_list", lambda: Dataset.from_polars(Dataset.from_dict({"l":[[1],[2,3]]}).to_polars()).features)
T("pl_fmt_row", lambda: Dataset.from_dict({"a":[1,2]}).with_format("polars")[0])
T("pl_fmt_col", lambda: Dataset.from_dict({"a":[1,2]}).with_format("polars")["a"])
T("pl_fmt_array2d", lambda: Dataset.from_dict({"x":[[[1,2],[3,4]]]}, features=Features({"x": Array2D((2,2),"int32")})).with_format("polars")[0])
# FASTA / FASTQ
open(td+"/a.fasta","w", encoding="utf-8-sig").write(">s1 desc\nACGT\n>s2\nGG\n")
T("fasta_bom", lambda: load_dataset("fasta", data_files=td+"/a.fasta", split="train", columns=["id","sequence"]).to_list())
open(td+"/b.fasta","w").write(">s1 desc\nAC GT\n\n;comment\nTT\n>\nGG\n>s3\n")
T("fasta_misc", lambda: load_dataset("fasta", data_files=td+"/b.fasta", split="train", columns=["id","description","sequence"]).to_list())
open(td+"/c.fasta","w", newline="").write(">s1\r\nACGT\r\nAA\r\n>s2\r\nGG\r\n")
T("fasta_crlf", lambda: load_dataset("fasta", data_files=td+"/c.fasta", split="train", columns=["id","sequence"]).to_list())
open(td+"/a.fastq","w", encoding="utf-8-sig").write("@r1 d\nACGT\n+\nIIII\n@r2\nGG\n+r2\n@@\n")
T("fastq_bom", lambda: load_dataset("fastq", data_files=td+"/a.fastq", split="train", columns=["id","sequence","quality"]).to_list())
open(td+"/b.fastq","w").write("@r1\nAC\nGT\n+\nII\nII\n@r2\n\n+\n\n@r3\nA\n+\n@\n")
T("fastq_multi", lambda: load_dataset("fastq", data_files=td+"/b.fastq", split="train", columns=["id","sequence","quality"]).to_list())
open(td+"/c.fastq","w").write("@r1 desc here\nACGT\n+r1\nIIII\n")
T("fastq_plus_id_only", lambda: load_dataset("fastq", data_files=td+"/c.fastq", split="train", columns=["id","sequence","quality"]).to_list())
T("text_bom", lambda: (open(td+"/t.txt","w",encoding="utf-8-sig").write("hello\nworld\n"), load_dataset("text", data_files=td+"/t.txt", split="train")["text"])[1])
T("json_bom", lambda: (open(td+"/bom.jsonl","w",encoding="utf-8-sig").write('{"a":1}\n{"a":2}\n'), load_dataset("json", data_files=td+"/bom.jsonl", split="train")["a"])[1])
T("csv_bom", lambda: (open(td+"/bom.csv","w",encoding="utf-8-sig").write('a,b\n1,2\n'), load_dataset("csv", data_files=td+"/bom.csv", split="train").column_names)[1])
