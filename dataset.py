import torch
from torch.utils.data import Dataset

class TwitterDataset(Dataset):

    def __init__(self, dataset, tokenizer_source, tokenizer_target, seq_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.seq_len = seq_len
        self.sos_token = torch.tensor([tokenizer_source.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_source.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_source.token_to_id('[PAD]')], dtype = torch.int64)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        source_target_pair = self.dataset[index]
        source_text = source_target_pair['text']
        target_text = str(source_target_pair['label'])

        encoder_input_tokens = self.tokenizer_source.encode(source_text).ids
        decoder_input_tokens = self.tokenizer_target.encode(target_text).ids

        encoder_num_padding_tokens = self.seq_len - len(encoder_input_tokens) - 2
        decoder_num_padding_tokens = self.seq_len - len(decoder_input_tokens) - 1

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Add SOS, EOS, and PAD tokens to encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # Add SOS and PAD tokens to decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype = torch.int64)
            ]
        )

        # Add EOS and PAD tokens to expected decoder output
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'source_text': source_text,
            'target_text': target_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0
