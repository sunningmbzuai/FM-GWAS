"""
Use permutation to compute p value
Support both Linear regression and logistic regression
Labels and covariants are from cohort file
feature name : (trait)_(cohort)_(sex)_test_predictions
save path: results_insample_10K/(trait)_(cohort)_(sex)_perm(perm)
Support F test for both linear and logistic regression
"""
import pandas as pd
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['VECLIB_MAXIMUM_THREADS'] = '2'

import math
import glob
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm
import torch
from sklearn.decomposition import PCA
from scipy.stats import truncnorm, norm
from scipy.stats import chi2
from scipy.stats import f as fdist
from time import perf_counter
from numpy.linalg import qr
import argparse
from numpy.linalg import solve
from os.path import exists, join

import pickle


"""
python insample_permutation_trait_MM_64K.py VAT M
python insample_permutation_trait_MM_64K.py Height F --cohort 10K
"""
def normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    mask = std != 0
    X_valid = X[:, mask]
    means_valid = mean[mask]
    stds_valid = std[mask]
    X_norm = (X_valid - means_valid) / stds_valid
    return X_norm


def compute_ll(model):
    return model.llf


def compute_lrt_stat(ll_full, ll_reduced):
    return 2 * (ll_full - ll_reduced)


def _qr_Q(X):
    Q, _ = qr(X, mode='reduced')
    return Q


def _ssr_via_Q(Q, y):
    t = Q.T @ y
    return float(y.T @ y - t.T @ t)


def _F_from_ssr(ssr_red, ssr_full, k, df2):
    return ((ssr_red - ssr_full) / k) / (ssr_full / df2)


def LinearRegression_Ftest(X, Z, y):
    y = np.asarray(y).reshape(-1)
    X = np.asarray(X); X = X if X.ndim == 2 else X.reshape(-1, 1)
    Z = np.asarray(Z); Z = Z if Z.ndim == 2 else Z.reshape(-1, 1)
    n = y.shape[0]
    assert X.shape[0] == n and Z.shape[0] == n, "X/Z must align with y"

    Zc = sm.add_constant(Z, has_constant='add')
    XZc = np.column_stack([Zc, X])
    k = X.shape[1]

    Q_red = _qr_Q(Zc)
    Q_full = _qr_Q(XZc)
    df2 = n - Q_full.shape[1]

    ssr_red_obs = _ssr_via_Q(Q_red, y)
    ssr_full_obs = _ssr_via_Q(Q_full, y)
    F_obs = _F_from_ssr(ssr_red_obs, ssr_full_obs, k, df2)
    p_F_analytic = float(fdist.sf(F_obs, k, df2))

    return {"F_obs": float(F_obs), "p_F_analytic": float(p_F_analytic)}



def align_by_ids(all_ids, feature_dict, keep_ids):
        """Subset and reorder arrays by keep_ids (dropping missing ones)."""
        id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
        valid_ids = [sid for sid in keep_ids if sid in id_to_idx]
        idx = [id_to_idx[sid] for sid in valid_ids]
        for k in feature_dict:
            feature_dict[k] = feature_dict[k][idx]
        return valid_ids, feature_dict


def load_features(win_paths, df_phenotype, sex=0, rebin_batch_size=256):
    """
    1. Filter participants by sex.
    2. Load .pt and .pkl features.
    3. Align everything (features, labels, covariates) by the same ordered subject_id.
    """
    # ---- Step 1: Filter phenotype by sex ----
    # sex could be 0/1 or F/M

    if isinstance(sex, str):
        sex = sex.upper()
        if sex == "F":
            sex = 0
        elif sex == "M":
            sex = 1
        elif sex == 'ALL':
            sex= -1
        else:
            raise ValueError("Invalid sex input. Use 0/1 or 'F'/'M'.")
    if sex == -1 :
        df = df_phenotype[df_phenotype["sex"].isin([0,1])].copy()
    else:
        df = df_phenotype[df_phenotype["sex"] == sex].copy()
    ordered_ids = df["participant_id"].astype(int).values
    print(f"Selected {len(ordered_ids)} participants with sex={sex}")

    # ---- Step 2: Load feature files ----
    pt_data =  {}
    pt_ids= None

    for path in win_paths:
        print(f"Loading {path}")
        if path.endswith(".pt"):
            data = torch.load(path, map_location="cpu")
            # if len(data['subject_id']) == 0:
            #     data1 = torch.load('/home/ec2-user/studies/modeling-hpp/ning/ModelGenerator/zeroshot_result_TransUnet_524K_snps_only/ENSG00000111537.4/ENSG00000111537.4.pt', map_location="cpu")
            #     data['subject_id'] = data1['subject_id']
            #     torch.save(data, path)

            pt_ids = np.array([int(s.split("_")[0]) for s in data["subject_id"]])
            cage_dim = 546
            rnaseq_dim = 667
            atac_dim=167
            dnase_dim=305

            # === mut2 only ===
            pt_data = {
                "concat_cage": np.array(data['concat_cage_mean']),
                "concat_rnaseq": np.array(data['concat_rna_seq_mean']),
                "concat_atac": np.array(data['concat_atac_mean']),
                "concat_dnase": np.array(data['concat_dnase_mean']),
                "concat_tss_cage": np.array(data['concat_cage_tss500_mean'].float()),
                "concat_tss_rnaseq": np.array(data['concat_rna_seq_tss500_mean'].float()),
                "concat_tss_atac": np.array(data['concat_atac_tss500_mean'].float()),
                "concat_tss_dnase": np.array(data['concat_dnase_tss500_mean'].float()),
                "add_cage": np.array(data['concat_cage_mean'])[..., :cage_dim] + np.array(data['concat_cage_mean'])[..., cage_dim:],
                "add_rnaseq": np.array(data['concat_rna_seq_mean'])[..., :rnaseq_dim] + np.array(data['concat_rna_seq_mean'])[..., rnaseq_dim:],
                "add_atac": np.array(data['concat_atac_mean'])[..., : atac_dim] + np.array(data['concat_atac_mean'])[...,atac_dim:],
                "add_dnase": np.array(data['concat_dnase_mean'])[..., : dnase_dim] + np.array(data['concat_dnase_mean'])[..., dnase_dim:],
                "add_tss_cage": np.array(data['concat_cage_tss500_mean'].float())[..., :cage_dim] + np.array(data['concat_cage_tss500_mean'].float())[..., cage_dim:],
                "add_tss_rnaseq": np.array(data['concat_rna_seq_tss500_mean'].float())[..., :rnaseq_dim] + np.array(data['concat_rna_seq_tss500_mean'].float())[..., rnaseq_dim:],
                "add_tss_atac": np.array(data['concat_atac_tss500_mean'].float())[..., : atac_dim] + np.array(data['concat_atac_tss500_mean'].float())[..., atac_dim:],
                "add_tss_dnase": np.array(data['concat_dnase_tss500_mean'].float())[..., : dnase_dim] + np.array(data['concat_dnase_tss500_mean'].float())[..., dnase_dim:]

            }

        else:
            raise ValueError(f"Unsupported file: {path}")
    # ---- Step 3: Align both feature sets to the same ordered subject_id ---

    # align .pt and .pkl
    if pt_ids is not None:
        ordered_ids, pt_data = align_by_ids(pt_ids, pt_data, ordered_ids)


    print(f"Aligned {len(ordered_ids)} participants with complete features.")

    # ---- Step 4: Subset phenotype to these aligned IDs ----
    df = df[df["participant_id"].isin(ordered_ids)].set_index("participant_id").loc[ordered_ids].reset_index()

    # ---- Step 5: Merge all aligned outputs ----
    feature_dict = {
        "subject_ids": ordered_ids,
        "labels": df["label"].values.reshape(-1, 1),
        "covariants": df[["age"]].values if sex >-1 else df[['age', 'sex']].values,
        **{f"{k}": v for k, v in pt_data.items()},
    }

    print(f"Final feature_dict contains {len(ordered_ids)} samples.")
    return feature_dict


def fix_signs(A):
    for i in range(A.shape[1]):
        if A[:, i].sum() < 0:
            A[:, i] *= -1
    return A


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process trait and sex input.")
    parser.add_argument("trait", type=str, help="Trait name (e.g., BMI, VAT)")
    parser.add_argument("sex", type=str, choices=["M", "F", "All"], help="Sex (M or F or All)")
    parser.add_argument("rank", type=int)
    parser.add_argument("--world_size", type=int, default=4)

    parser.add_argument("--cohort", type=str, default="extreme", choices=["extreme", "10K", "1K"], help="cohort (extreme,1K)")
    parser.add_argument("--overwrite", action="store_true", help="If set, existing files will be overwritten.")
    parser.add_argument("--reverse", action="store_true", help="If set, reverse file order")
    parser.add_argument(
        "--feature_root", type=str, required=True,
        help="Root dir of gene embeddings, i.e. --output_path passed to "
             "get_hpp_embedding-AG_clean.py (contains <gene_id>/<gene_id>.pt).",
    )
    parser.add_argument(
        "--gene_file", type=str, required=True,
        help="Candidate gene list tsv with a gene_id column, e.g. "
             "HPP_union_gene/{trait}_extreme.tsv (or _F/_M for sex-stratified).",
    )
    parser.add_argument(
        "--cohort_file", type=str, required=True,
        help="Cohort tsv for this trait with participant_id, sex, age, and "
             "label (the trait phenotype) columns.",
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Output directory for per-gene association result tsv files.",
    )

    args = parser.parse_args()

    trait_name = args.trait
    cohort = args.cohort  # extreme, 10K, extreme_2K
    sex_flag = args.sex  # F, M, All

    feature_root = args.feature_root
    gene_file = args.gene_file
    cohort_file = args.cohort_file
    save_path = args.save_path

    df_gene = pd.read_csv(gene_file, sep='\t')
    os.makedirs(save_path, exist_ok=True)

    print(f'Running permutation insample analysis')
    print(f'Trait: {trait_name}')
    print(f'Cohort {cohort}')
    print(f'Sex: {sex_flag}')
    print(f'feature root: {feature_root}')
    print(f'save path: {save_path}')

    n = len(df_gene)
    indices = np.arange(n)

    my_indices = indices[indices % args.world_size == args.rank]
    df_rank = df_gene.iloc[my_indices]

    for i, row  in tqdm(df_rank.iterrows(), total=len(df_rank)):
        gene_id = row['gene_id']

        # skip finished genes
        if exists(join(save_path, gene_id+'.tsv')) and not args.overwrite:
            continue
        df_phenotype = pd.read_csv(cohort_file, sep='\t')
        is_binary = df_phenotype['label'].nunique() == 2
        # load feature paths, including .pt and .pkl
        win_paths = glob.glob(join(feature_root, gene_id, f'{gene_id}.pt'))

        if len(win_paths) < 1:
            print(f'{gene_id} not finished, skip')
            continue
        # load features
        data = load_features(win_paths, df_phenotype, sex=args.sex)

        Z = np.asarray(data['covariants'], dtype=np.float64)
        labels = data['labels']
        # Covariates enter OLS directly (no PCA). Previously PCA(0.99) on raw [age, sex]
        # kept only PC1 (~age, since var(age) >> var(sex)), silently dropping sex for sex=All.
        expected_cov = ['age', 'sex'] if args.sex.upper() == 'ALL' else ['age']
        if Z.ndim != 2 or Z.shape[1] != len(expected_cov):
            raise ValueError(f'Expected covariates {expected_cov}, got shape {Z.shape}')
        cov_std = Z.std(axis=0)
        if np.any(cov_std == 0):
            raise ValueError(f'Constant covariate(s) {[c for c, s in zip(expected_cov, cov_std) if s == 0]} '
                             f'for sex={args.sex}')
        save_dict = {'gene_id': [], 'num_feature': [], 'embed_type': [], 'F_obs': [], 'p_F_analytic': []}

        if not is_binary:  # Use Linear regression
            print(f'Trait {trait_name}, Use Linear Regression')
            Y = normalization(labels)
        else:
            print(f'Trait {trait_name}, Use Logistic Regression')
            Y = labels

        embed_types = []

        for track_type in ['cage', 'rnaseq', 'dnase', 'atac', 'rnaseq_cage', 'rnaseq_dnase', 'rnaseq_atac', 'rnaseq_cage_dnase', 'cage_dnase', 'cage_atac', 'dnase_atac', 'rnaseq_cage_atac', 'rnaseq_dnase_atac', 'cage_dnase_atac', 'rnaseq_cage_dnase_atac']:
            for feat_type in ['concat', 'add', 'concat_tss', 'add_tss']:
            # for feat_type in ['concat', 'add']:
                embed_types.append(f'{feat_type}_{track_type}')
        for embed_type in embed_types:
            # Gene PCA: keep 95% variance, whitened, sklearn 'auto' solver (same config for all embed types)
            pca_component_list = [0.95]
            for pca_component in pca_component_list:
                if embed_type in data.keys():
                    X = data[embed_type]
                else: # use more than one track type
                    if 'tss' in embed_type:
                        feat_names = [t for t in embed_type.split('_')[2:]]
                        prefix = embed_type.split('_')[0] + '_tss'
                    else:
                        feat_names = [t for t in embed_type.split('_')[1:]]
                        prefix = embed_type.split('_')[0]

                    X = np.concatenate([data[f'{prefix}_{t}'] for t in feat_names], axis=-1)
                    # print(embed_type, X.shape)


                if np.isnan(X).any():
                    print(f"{gene_id} has NaN, skipping...")
                    break
                if X.size==0:
                    print(f'{embed_type} does not have embedding, break')
                    break
                # gene embedding PCA
                X = normalization(X.astype(np.float32))
                if X.shape[-1] == 0:
                    print(f'{embed_type}: all features constant, skip')
                    break
                pca_final = PCA(n_components=pca_component, whiten=True, svd_solver='auto')
                try:
                    X = pca_final.fit_transform(X)
                except Exception as e:
                    print(f'{gene_id} {embed_type}: PCA failed ({e}), skip')
                    continue
                # fix sign to ensure determistic
                X = fix_signs(X)
                # save pval
                ret = LinearRegression_Ftest(X, Z, Y)
                # print(embed_type+f'_pca{pca_component}', ret['p_F_analytic'])
                save_dict['gene_id'].append(gene_id)
                save_dict['num_feature'].append(X.shape[-1])
                save_dict['embed_type'].append(embed_type+f'_pca{pca_component}')
                for k, v in ret.items():
                    save_dict[k].append(v)

        df = pd.DataFrame.from_dict(save_dict)
        save_file = os.path.join(save_path, f'{gene_id}.tsv')
        df.to_csv(save_file, sep='\t', index=False)
        print(f'save to {save_file}')
