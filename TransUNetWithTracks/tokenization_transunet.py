"""Tokenization classes for TransUNet model.
Currently supports only protein
"""

import os
from typing import List, Optional

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

import numpy as np
import torch
import random


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "transunet/proteinmoe": "vocab.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"proteinmoe": 2048, "rnabert": 1024}


def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [l.rstrip('\n') for l in lines]

def make_a_sample(  sequence, tokenizer,
                    mask_rate: Optional[float] = None,
                    padding_to: Optional[int] = None,
                    max_length: Optional[int] = None,
                    add_begin_end_token: bool = False,
                    make_input_upper_case: bool = False,
                    make_label_upper_case: bool = False):
    """
    Make a sample for model
    
    Parameters
    ------------------
    sequence: e.g. "T T G A A A A A T G A T"
    tokenizer: tokenizer from mm_utils.get_tokenizer
    mask_rate: float or None.
    padding_to: int. Padding to specific length
    max_length: int. limit the maximun length to this value if provide
    add_begin_end_token: add [bos] and [eos] to the both end of sequence
    make_input_upper_case: bool. Input will be converted to upper-case
    make_label_upper_case: bool. Label will be converted to upper-case
    
    Return
    -----------------
    batch: dict
        - tokens: [L]
        - labels: [L]
        - attention_mask: [1, 1, L]
        - loss_mask: [L], 2 for masked tokens
        - position_ids: [L]
        - padding_mask: [L], 1 for padding tokens
        - special_mask: [L], 1 for BOS and EOS tokens
    """
    
    assert isinstance(sequence, str) and " " in sequence, f"tokens should be seperated by space: {sequence}"
    PAD_TOKEN = '[pad]' if '[pad]' in tokenizer.all_tokens else '[PAD]'
    assert PAD_TOKEN in tokenizer.all_tokens
    assert all([x in tokenizer.all_tokens for x in 'ATCG'])
    BOS_TOKEN = '[BOS]' if '[BOS]' in tokenizer.all_tokens else '<bos>'
    EOS_TOKEN = '[EOS]' if '[EOS]' in tokenizer.all_tokens else '<eos>'
    assert BOS_TOKEN in tokenizer.all_tokens and EOS_TOKEN in tokenizer.all_tokens
    MASK_TOKEN = '[MASK]'
    assert MASK_TOKEN in tokenizer.all_tokens
    UNKNOWN_TOKEN = 'N'
    assert UNKNOWN_TOKEN in tokenizer.all_tokens
    TOKENIZER_SUPPORT_LOWERCASE = ('a' in tokenizer.all_tokens)
    
    tokens = sequence.split()
    if not TOKENIZER_SUPPORT_LOWERCASE:
        #assert make_input_upper_case, f"tokenizer not support lower-case letters"
        #assert make_label_upper_case, f"tokenizer not support lower-case letters"
        tokens = [ t.upper() if len(t)==1 else t for t in tokens ]
    
    special_mask = [0] * len(tokens)
    if add_begin_end_token:
        tokens = [BOS_TOKEN] + tokens + [EOS_TOKEN]
        special_mask = [1] + special_mask + [1]
    
    if padding_to is not None and max_length is not None:
        assert padding_to <= max_length, f"expect padding_to <= max_length"
    
    if padding_to is not None and len(tokens) < padding_to:
        padding_mask = np.zeros([padding_to], dtype=np.int_)
        padding_mask[len(tokens):] = 1
        special_mask += [0] * (padding_to - len(tokens))
        tokens += [PAD_TOKEN] * (padding_to - len(tokens))
        assert len(tokens) == padding_to
    elif max_length is not None and len(tokens) > max_length:
        if add_begin_end_token:
            tokens = tokens[:max_length-1] + [EOS_TOKEN]
            special_mask = special_mask[:max_length-1] + [1]
        else:
            tokens = tokens[:max_length]
            special_mask = special_mask[:max_length]
        padding_mask = np.zeros([len(tokens)], dtype=np.int_)
    else:
        padding_mask = np.zeros([len(tokens)], dtype=np.int_)
    
    unknown_token_id = tokenizer._token_to_id[UNKNOWN_TOKEN]
    labels = np.array([ tokenizer._token_to_id.get(x, unknown_token_id) for x in tokens ])
    tokens = labels.copy()
    loss_mask = np.zeros(len(tokens), dtype=np.int_)
    special_mask = np.array(special_mask)
    attention_mask = np.zeros([1, 1, len(tokens)]).astype(np.bool_)
    position_ids = np.arange(0, len(tokens))
    
    if mask_rate is not None and mask_rate > 0:
        valid_token_ids = [ tokenizer._token_to_id[x] for x in 'ATCG' ]
        if TOKENIZER_SUPPORT_LOWERCASE:
            valid_token_ids.extend([ tokenizer._token_to_id[x] for x in 'atcg' ])
        index = np.arange(0, len(tokens))[ padding_mask==0 ].tolist()
        random.Random(0).shuffle(index)
        index = np.array(sorted(index[:int(len(index)*mask_rate)]))
        try:
            index = index[ np.isin(labels[index], valid_token_ids) ]
        except:
            breakpoint()
        loss_mask[index] = 1.0
        tokens[index] = tokenizer._token_to_id[MASK_TOKEN]
    
    ### Lower-case to upper-case
    if TOKENIZER_SUPPORT_LOWERCASE:
        upper_case_mapping = {'a': 'A', 't': 'T', 'c': 'C', 'g': 'G', 'n': 'N'}
        upper_case_mapping = { tokenizer._token_to_id[k]:tokenizer._token_to_id[v] for k,v in upper_case_mapping.items() }
        make_upper_case_func = np.vectorize(lambda x: upper_case_mapping.get(x, x))
        if make_input_upper_case:
            tokens = make_upper_case_func(tokens)
        if make_label_upper_case:
            labels = make_upper_case_func(labels)
    
    batch = {
        'input_ids': tokens,
        'labels': labels,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
        'padding_mask': padding_mask,
        'special_mask': special_mask,
        'position_ids': position_ids,
    }
    
    batch = { k:torch.as_tensor(v) for k,v in batch.items() }
    
    return batch

class TransUNetTokenizer(PreTrainedTokenizer):
    """
    Constructs a TransUNet tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        biotype="protein",
        unk_token="-",
        pad_token="[PAD]",
        mask_token="[MASK]",
        sep_token="[SEP]",
        cls_token=None,
        bos_token=None,
        eos_token=None,
        **kwargs,
    ):
        """
        Args:
            biotype: str, could be protein/rna/dna
            the input is like ...[SEP]
        """
        self.biotype = biotype
        if self.biotype != "protein":
            raise NotImplementedError

        self.all_tokens = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}

        super().__init__(
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sep_token=sep_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

        # TODO, all the tokens are added? But they are also part of the vocab... bit strange.
        # none of them are special, but they all need special splitting.

        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text, **kwargs):
        """
        Hack for multiple chains (seperated by |)
        Args:
            text: str, eg. CALVSGGNYKPTF|CASSWGGAPLF|ELAGIGILTV
        """
        return text.replace("|", self.sep_token).split()

    def get_vocab(self):
        base_vocab = self._token_to_id.copy()
        base_vocab.update(self.added_tokens_encoder)
        return base_vocab

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    @property
    def token2id(self) -> dict:
        return self._token_to_id
    
    @property
    def id2token(self) -> dict:
        return self._id_to_token

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if self.biotype == "protein":
            sep = [self.sep_token_id]
            if token_ids_1 is None:
                return token_ids_0 + sep
            else:
                return token_ids_0 + sep + token_ids_1 + sep
        else:
            raise NotImplementedError

    def get_special_tokens_mask(
        self,
        token_ids_0: List,
        token_ids_1: Optional[List] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory, filename_prefix):
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt",
        )
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return len(self.all_tokens)

    def make_a_batch(
        self,
        texts,
        mask_rate: Optional[float] = None,
        padding: Optional[object] = "longest",
        max_length: Optional[int] = None,
        add_begin_end_token: bool = False,
        make_input_upper_case: bool = False,
        make_label_upper_case: bool = False,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ):
        """
        Build model inputs by delegating per-sample work to make_a_sample, then collate.
        - texts: str or List[str] (e.g. "AAAATAGTC" or ["AAA", "TTT"])
        - Inserts [SEP] between chains if input contains '|'
        - padding: True/"longest"/int/False (default "longest")
        """
        # Normalize inputs
        if isinstance(texts, str):
            texts_list = [texts]
            single = True
        else:
            texts_list = list(texts)
            single = False

        spaced_texts = [ " ".join(list(seq)) for seq in texts_list ]
        pad_to = None
        if padding in (True, "longest"):
            pad_to = max([len(seq) for seq in texts_list]) + (2 if add_begin_end_token else 0)
        elif isinstance(padding, int):
            pad_to = padding

        # Build samples
        samples = [
            make_a_sample(
                sequence=seq,
                tokenizer=self,
                mask_rate=mask_rate,
                padding_to=pad_to,
                max_length=max_length,
                add_begin_end_token=add_begin_end_token,
                make_input_upper_case=make_input_upper_case,
                make_label_upper_case=make_label_upper_case,
            )
            for seq in spaced_texts
        ]

        if single:
            sample = samples[0]
            return sample

        # Collate to batch
        keys = samples[0].keys()
        batch = {}
        for k in keys:
            vals = [s[k] for s in samples]
            batch[k] = torch.stack(vals, dim=0)
        return batch
