import argparse,hashlib,json
from openpyxl import load_workbook
share=0.15
ap=argparse.ArgumentParser()
ap.add_argument("--all",action="store_true",help="train on every row, write no check set")
ap.add_argument("--boost-human",type=int,default=1,metavar="N",
help="write each medium + needs_human true row N times in train_column.jsonl (default 1)")
args=ap.parse_args()
rows=[]
with open("data/train_raw.jsonl",encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
print("train_raw.jsonl:",len(rows),"rows")
ws=load_workbook("data/data_starter.xlsx",data_only=True)["data_starter"]
data=list(ws.iter_rows(values_only=True))
header=data[0]
cnt=0
for values in data[1:]:
    if values[0] is None:
        continue
    r=dict(zip(header,values))
    r["needs_human"]=str(r["needs_human"]).strip().lower()=="true"
    rows.append(r)
    cnt+=1
print("data_starter.xlsx:",cnt,"rows")
print("total:",len(rows),"rows")
def chk(row):
    h=int(hashlib.sha1(str(row["id"]).encode("utf-8")).hexdigest()[:8],16)
    return h/0xFFFFFFFF<share
if args.all:
    train,check=rows,[]
else :
    train=[r for r in rows if not chk(r)]
    check=[r for r in rows if chk(r)]
print("training rows:",len(train),"  check set rows:",len(check))
cols=["id","message","category","urgency","needs_human"]
with open("data/train_merged.jsonl","w",encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps({k:r[k] for k in cols},ensure_ascii=False)+"\n")
if check:
    with open("data/check_set.jsonl","w",encoding="utf-8") as f:
        for r in check:
            f.write(json.dumps({k:r[k] for k in cols},ensure_ascii=False)+"\n")
sp=open("data/system_prompt.txt",encoding="utf-8").read().strip()
out=[]
for r in train:
    copies=args.boost_human if (r["urgency"]=="medium" and r["needs_human"]) else 1
    out.extend([r]*copies)
if args.boost_human>1:
    import random
    random.Random(0).shuffle(out)
    print(f"boost: medium + needs_human rows x{args.boost_human} -> train_column.jsonl has {len(out)} lines")
with open("data/train_column.jsonl","w",encoding="utf-8") as f:
    for r in out:
        a=json.dumps({"category":r["category"],
        "urgency":r["urgency"],
        "needs_human":r["needs_human"]})
        f.write(json.dumps({"system":sp,
        "user":r["message"],
        "assistant":a},ensure_ascii=False)+"\n")
print("wrote data/train_merged.jsonl and data/train_column.jsonl"+(" and data/check_set.jsonl" if check else ""))
