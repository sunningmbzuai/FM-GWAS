import argparse
import os
import re
import sys
import time
from os.path import basename, exists, join

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def dynamic_batch_size(length, gpu="A100"):
    multiple = 2 if gpu == "A100" else 1
    if length <= 8192:
        return 16 * multiple
    if length <= 25344:
        return 8 * multiple
    if length <= 51_712:
        return 4 * multiple
    if length <= 124672:
        return 2 * multiple
    if length <= 197888:
        return 1 * multiple
    if length >= 300_000:
        return 1
    return 2


class PairedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, padding_length=524288):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.padding_length = padding_length

        self.ref_key = "ref_seq"
        base_key = (
            "mutation_seq" if "mutation_seq" in self.df.columns
            else "mutated_seq" if "mutated_seq" in self.df.columns
            else "variant_seq"
        )
        self.mut1_key = f"{base_key}_1"
        self.mut2_key = f"{base_key}_2"
        self.subject_key = (
            "participant_id" if "participant_id" in self.df.columns
            else "subject_id" if "subject_id" in self.df.columns
            else "dosage" if "dosage" in self.df.columns
            else None
        )
        if self.subject_key is None:
            raise ValueError("Expected one of participant_id, subject_id, or dosage in input TSV.")

        max_seq_len = max(
            self.df[self.ref_key].str.len().max(),
            self.df[self.mut1_key].str.len().max(),
            self.df[self.mut2_key].str.len().max(),
        )
        if max_seq_len % 256:
            max_seq_len = ((max_seq_len // 256) + 1) * 256
        self.max_seq_len = min(int(max_seq_len), self.padding_length)
        print(f"real max length {self.max_seq_len}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        out = {
            "subject_id": row[self.subject_key],
            self.ref_key: row[self.ref_key],
            self.mut1_key: row[self.mut1_key],
            self.mut2_key: row[self.mut2_key],
        }

        for prefix, key in (("ref", self.ref_key), ("mut1", self.mut1_key), ("mut2", self.mut2_key)):
            seq = " ".join(row[key])
            batch = self.tokenizer.make_a_batch(
                seq, padding=self.max_seq_len, max_length=self.max_seq_len
            )
            out.update({f"{prefix}_{k}": v for k, v in batch.items()})
        return out


def get_batch_embeddings(model, batch, prefix="mut1"):
    batch = {
        k: (v.cuda() if torch.is_tensor(v) else v)
        for k, v in batch.items()
        if k.startswith(prefix)
    }
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False), torch.no_grad():
        output = model.predict_tracks(batch[f"{prefix}_input_ids"], is_human=True)

    mask = (
        (batch[f"{prefix}_special_mask"] == 0)
        & (batch[f"{prefix}_padding_mask"] == 0)
    ).float()
    return output, mask


def masked_mean_pooling(embed, mask, eps=1e-8):
    mask = mask.unsqueeze(-1)
    return (embed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=eps)


def get_track_values(output, track_name):
    if track_name == "cage":
        return output.cage.values
    if track_name == "rna_seq":
        return output.rna_seq.values
    if track_name == "atac":
        return output.atac.values
    if track_name == "dnase":
        return output.dnase.values
    raise ValueError(f"track name {track_name} does not support")


def get_track_mean(mut1, mut2, mask1, mask2, track_name, tss_window=500):
    mut1_track = get_track_values(mut1, track_name)
    mut2_track = get_track_values(mut2, track_name)

    mut1_mean = masked_mean_pooling(mut1_track, mask1)
    mut2_mean = masked_mean_pooling(mut2_track, mask2)
    mut1_tss = mut1_track[:, :tss_window].mean(dim=1)
    mut2_tss = mut2_track[:, :tss_window].mean(dim=1)
    return torch.cat([mut1_mean, mut2_mean], dim=-1), torch.cat([mut1_tss, mut2_tss], dim=-1)


def get_dataset_embeddings(dataset, model, batch_size=32, num_workers=4, save_file="gene.pt"):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    for i, batch in enumerate(dataloader):
        _ = get_batch_embeddings(model, batch, prefix="ref")
        if i >= 2:
            break

    track_list = ["cage", "rna_seq", "atac", "dnase"]
    track_embed_dict = {k: [] for k in track_list}
    tss_embed_dict = {k: [] for k in track_list}
    subject_ids = []

    for batch in tqdm(dataloader, dynamic_ncols=True, leave=False):
        mut1_output, mut1_masks = get_batch_embeddings(model, batch, prefix="mut1")
        mut2_output, mut2_masks = get_batch_embeddings(model, batch, prefix="mut2")

        for track_name in track_list:
            track_embed, tss_track_embed = get_track_mean(
                mut1_output,
                mut2_output,
                mut1_masks,
                mut2_masks,
                track_name=track_name,
                tss_window=500,
            )
            track_embed_dict[track_name].append(track_embed)
            tss_embed_dict[track_name].append(tss_track_embed)

        subject_ids.extend(batch["subject_id"])

    output_dict = {
        "subject_id": subject_ids,
        "concat_cage_mean": torch.cat(track_embed_dict["cage"], dim=0),
        "concat_rna_seq_mean": torch.cat(track_embed_dict["rna_seq"], dim=0),
        "concat_atac_mean": torch.cat(track_embed_dict["atac"], dim=0),
        "concat_dnase_mean": torch.cat(track_embed_dict["dnase"], dim=0),
        "concat_cage_tss500_mean": torch.cat(tss_embed_dict["cage"], dim=0),
        "concat_rna_seq_tss500_mean": torch.cat(tss_embed_dict["rna_seq"], dim=0),
        "concat_atac_tss500_mean": torch.cat(tss_embed_dict["atac"], dim=0),
        "concat_dnase_tss500_mean": torch.cat(tss_embed_dict["dnase"], dim=0),
    }

    torch.save(output_dict, save_file)
    print(f"Saved -> {save_file}")


def initialize_model(model_path, compile_model, compiled_binary=None, compiled_binary_max_length=None):
    if os.path.exists(model_path):
        ckpt_path = model_path
    else:
        from huggingface_hub import snapshot_download

        ckpt_path = snapshot_download(repo_id=model_path)

    print(f"Loading model from {ckpt_path}, compile={compile_model}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if compiled_binary is not None:
        # Load a pre-compiled AOTInductor .pt2 package instead of the HF
        # checkpoint, e.g. TransUNetWithTracks/compiled_A100/aido_dna3ag_1bp4_L8192_dynbatch_a100.pt2
        # or .../aido_dna3ag_1bp4_L128000_dynbatch_a100.pt2 (A100/sm_80), or
        # TransUNetWithTracks/compiled/aido_dna3ag_1bp4_dyn.pt2 (H100/sm_90).
        # Exposes the same .predict_tracks(input_ids, is_human=True) interface used below.
        print(f"Loading compiled AOTInductor binary from {compiled_binary}")
        binary_dir = os.path.dirname(os.path.abspath(compiled_binary))
        if binary_dir not in sys.path:
            sys.path.insert(0, binary_dir)
        import aido_dna3ag_binary

        max_length = compiled_binary_max_length
        basename = os.path.basename(compiled_binary)
        if max_length is None and "_dyn_a100" in basename:
            # aido_dna3ag_1bp4_dyn_a100.pt2: genuinely dynamic build (batch
            # 1-32, length any multiple of 128 in [256, 128000]) -- the
            # wrapper auto-detects its shape support from the filename.
            print("Compiled binary: dynamic length (multiple of 128, up to 128000), dynamic batch (up to 32)")
            model = aido_dna3ag_binary.load(compiled_binary).cuda().eval()
            return model, tokenizer
        if max_length is None:
            # The fixed-shape A100 builds are named "..._L<length>_...pt2" --
            # a fixed internal sequence length baked in at compile time (see
            # compiled_A100/README.md). Recover it from the filename so the
            # wrapper knows how far it can right-pad shorter inputs.
            m = re.search(r"_L(\d+)_", basename)
            if not m:
                raise ValueError(
                    "Could not infer the compiled binary's fixed sequence length "
                    f"from its filename ({compiled_binary}); pass "
                    "--compiled_binary_max_length explicitly."
                )
            max_length = int(m.group(1))
        print(f"Compiled binary fixed sequence length: {max_length}")

        model = aido_dna3ag_binary.load(compiled_binary, max_length=max_length).cuda().eval()
        return model, tokenizer

    print("start to load model weights ...")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda().eval()
    if compile_model:
        model = torch.compile(model)
    return model, tokenizer


def save_embeddings(
    model,
    tokenizer,
    input_path,
    output_path,
    padding_length,
    num_workers,
    overwrite,
    df_gene,
    gpu_type,
):
    saved, skipped = 0, 0
    for _, row in df_gene.iterrows():
        gene_id = row["gene_id"]

        win_file = join(input_path, f"{gene_id}.tsv")
        if not os.path.exists(win_file):
            print(f"not found {win_file}")
            continue

        out_gene_dir = join(output_path, gene_id)
        os.makedirs(out_gene_dir, exist_ok=True)
        out_pt = join(out_gene_dir, basename(win_file).replace(".tsv", ".pt"))

        if not overwrite and exists(out_pt):
            skipped += 1
            continue

        print(f"Processing {win_file} -> {out_pt}")
        df = pd.read_csv(win_file, sep="\t")
        if len(df) == 0:
            print(f"Empty dataframe: {win_file}, skip")
            skipped += 1
            continue

        dataset = PairedSequenceDataset(df, tokenizer, padding_length=padding_length)
        print(f"  Dataset size: {len(dataset)}, max_seq_len: {dataset.max_seq_len}")
        batch_size = dynamic_batch_size(dataset.max_seq_len, gpu_type)

        get_dataset_embeddings(
            dataset,
            model,
            batch_size=batch_size,
            num_workers=num_workers,
            save_file=out_pt,
        )
        saved += 1
        torch.cuda.empty_cache()

    print(f"Done. Saved: {saved}, Skipped: {skipped}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ec2-user/studies/modeling-hpp/ning/TransUnet-837M-524K-AG",
        help="Path to model checkpoint directory or HF repo",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/home/ec2-user/studies/modeling-hpp/ning/processed_data_vcf/plink_snps_only",
        help="Path to data directory",
    )
    parser.add_argument("--output_path", type=str, help="Root dir to write outputs")
    parser.add_argument("--padding_length", type=int, default=524288)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--compiled_binary",
        type=str,
        default=None,
        help="Path to a pre-compiled AOTInductor .pt2 package (e.g. "
        "TransUNetWithTracks/compiled_A100/aido_dna3ag_1bp4_L8192_dynbatch_a100.pt2) "
        "to use instead of loading the HF checkpoint via trust_remote_code.",
    )
    parser.add_argument(
        "--compiled_binary_max_length",
        type=int,
        default=None,
        help="Fixed internal sequence length of --compiled_binary. Inferred from "
        "the filename (the '_L<length>_' part) if not given.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--gene_annotation_file",
        type=str,
        default="/home/ec2-user/studies/modeling-hpp/ning/raw_data/protein_coding_genes_valid.tsv",
    )
    parser.add_argument("--gpu", type=str, default="A100")
    return parser.parse_known_args()[0]


def main():
    args = parse_args()

    df_gene = pd.read_csv(args.gene_annotation_file, sep="\t")
    sub_df = df_gene.iloc[
        [i for i in range(len(df_gene)) if i % args.world_size == args.rank]
    ].reset_index(drop=True)
    print(f"[Rank {args.rank}] got {len(sub_df)}/{len(df_gene)} genes")

    model, tokenizer = initialize_model(
        args.model_path, args.compile, args.compiled_binary, args.compiled_binary_max_length
    )
    save_embeddings(
        model,
        tokenizer,
        args.input_path,
        args.output_path,
        args.padding_length,
        args.num_workers,
        args.overwrite,
        df_gene=sub_df,
        gpu_type=args.gpu,
    )


if __name__ == "__main__":
    main()
