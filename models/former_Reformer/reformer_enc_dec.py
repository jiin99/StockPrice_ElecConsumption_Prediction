import re
from torch import nn
from reformer_pytorch.reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

ENC_PREFIX = 'enc_'
DEC_PREFIX = 'dec_'

def group_dict_by_key(cond, d):
    # key에 대해서 cond에 넣어진 함수를 사용해서 검색. match된건 1 딕셔너리에, match되지 않은건 0 딕셔너리에 저장한다.
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    # 해당 string이 문자열의 처음부터 정규식과 매치되는지 조사하기 위해 match를 사용한다. 
    return bool(re.match(f'^{prefix}', str))

def group_by_key_prefix(prefix, d):
    # 딕셔너리의 Key값을 조사: key에 대해 prefix값을 검사
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    # x가 prefix값으로 시작되는지 확인하는 함수를 인수로 넣는다.
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def extract_enc_dec_kwargs(kwargs):
    # Encoder와 Decoder의 Argparse를 구분해서 반환(key에 대해서 PREFIX를 검색하고, 존재하는 것들을 반환. PREFIX값은 삭제된다. )
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs

def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'input_mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['input_mask'])
    return enc_kwargs, dec_kwargs, kwargs



class ReformerEncDec(nn.Module):
    def __init__(self, dim, ignore_index = 0, pad_value = 0, **kwargs):
        super().__init__()
        
        # 받은 인수들을 encoder용, decoder용으로 분류한다.
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'return_embedding' not in enc_kwargs, 'you cannot manually set the return embeddings flag for the encoder'
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['return_embeddings'] = True
        dec_kwargs['causal'] = True
        
        # bucket_size가 없을 경우, 인코더의 bucket_size 키값에 64를 넣는다. 디코더는 이의 두배값을 넣는다.
        enc_kwargs.setdefault('bucket_size', 64)
        dec_kwargs.setdefault('bucket_size', enc_kwargs['bucket_size'] * 2)

        ##파라미터 전달,인코더 디코더 Reformer 구조 사용해서 만든다.
        enc = ReformerLM(**enc_kwargs)
        dec = ReformerLM(**dec_kwargs)
        
        # Training wrapper사용..?
        self.enc = TrainingWrapper(enc, ignore_index = ignore_index, pad_value = pad_value)
        self.dec = TrainingWrapper(dec, ignore_index = ignore_index, pad_value = pad_value)

    def generate(self, seq_in, seq_out_start, seq_len, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        enc_keys = self.enc(seq_in, **enc_kwargs)
        return self.dec.generate(seq_out_start, seq_len, keys = enc_keys, **{**dec_kwargs, **kwargs})

    def forward(self, seq_in, seq_out, return_loss = False, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        enc_keys = self.enc(seq_in, **enc_kwargs)
        return self.dec(seq_out, return_loss = return_loss, keys = enc_keys, **dec_kwargs)
