#!/usr/bin/env python3
"""修 InternLM2 系(InternVL2.5-8B / InternVideo2.5-8B)的 sentencepiece 兼容问题。

症状:
    RuntimeError: INTERNAL: piece must not include null character.
原因:
    这两个模型的 tokenizer.model 里有**恰好一个** piece 的内容是字面量 NUL(`'\\x00'`,idx=354,
    type=1 NORMAL)。sentencepiece **0.2.0 起**加了严格校验,直接拒绝加载。
    而 Python 3.12 只有 0.2.0+ 的轮子(0.1.99 没有 cp312),没法靠降版本绕过。
修法:
    把这个 piece 的**内容**改名成一个正常文本里不可能出现的占位符,类型保持不变。
    - 词表大小不变、**其余所有 token id 一律不动**(只有 354 号这一个 piece 的字面量变了)
    - 字节回退 piece `<0x00>`(idx=3, type=6 BYTE)**原样保留**,真要编码 NUL 还是走它
    - 我们的 prompt 全是 ASCII 的 MCQ 文本,永远不会产生字面量 NUL token
      (脚本末尾会对全部 7,922 道题的 prompt 做一次实证核对)
原文件备份为 tokenizer.model.orig,可随时还原。
"""
import argparse, os, sys

PLACEHOLDER = "<unused_nul_piece>"


def patch(path, dry=False):
    from sentencepiece import sentencepiece_model_pb2 as pb
    raw = open(path, "rb").read()
    m = pb.ModelProto()
    m.ParseFromString(raw)
    bad = [i for i, p in enumerate(m.pieces) if "\x00" in p.piece]
    if not bad:
        print(f"  {path}: 没有含 NUL 的 piece,不用动")
        return False
    existing = {p.piece for p in m.pieces}
    for i in bad:
        ph = PLACEHOLDER
        n = 0
        while ph in existing:
            n += 1; ph = f"{PLACEHOLDER[:-1]}_{n}>"
        print(f"  idx={i} type={m.pieces[i].type} {m.pieces[i].piece!r} -> {ph!r}")
        m.pieces[i].piece = ph
        existing.add(ph)
    if dry:
        return True
    bak = path + ".orig"
    if not os.path.exists(bak):
        open(bak, "wb").write(raw)
        print(f"  原文件已备份 -> {bak}")
    open(path, "wb").write(m.SerializeToString())
    print(f"  已写回 {path}")
    return True


def verify(model_dir, pkg):
    """实证:用修好的 tokenizer 把全部题目的 prompt 编一遍,确认没有任何一条用到被改的 piece。"""
    sys.path.insert(0, os.path.join(pkg, "tools"))
    from common import build_prompt, load_split
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    bad_id = tok.convert_tokens_to_ids(PLACEHOLDER)
    items = load_split(pkg, "heldout_test,test")
    hits = 0
    for it in items:
        ids = tok(build_prompt(it, "plain"))["input_ids"]
        if bad_id is not None and bad_id in ids:
            hits += 1
    print(f"  核对 {len(items)} 条 prompt:用到被改 piece 的有 {hits} 条"
          f" {'✅ 零影响' if hits == 0 else '❌ 有影响,需重新评估'}")
    return hits == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="模型目录")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="补跑一次全量 prompt 核对")
    ap.add_argument("--pkg", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    for d in a.dirs:
        f = os.path.join(d, "tokenizer.model")
        if not os.path.exists(f):
            print(f"{d}: 没有 tokenizer.model,跳过"); continue
        print(f"=== {d} ===")
        changed = patch(f, a.dry_run)
        if changed and a.verify and not a.dry_run:
            verify(d, a.pkg)
