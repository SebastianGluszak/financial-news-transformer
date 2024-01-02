import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import TwitterDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path
from tqdm import tqdm

def get_all_data_type(dataset, data_type):
    for item in dataset:
        yield str(item[data_type])

def get_tokenizer(config, dataset, data_type):
    tokenizer_path = Path(config['tokenizer_file'].format(data_type))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_data_type(dataset, data_type), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    train_dataset_raw = load_dataset('zeroshot/twitter-financial-news-sentiment', split = 'train')
    validation_dataset_raw = load_dataset('zeroshot/twitter-financial-news-sentiment', split = 'validation')

    # Build tokenizers
    tokenizer_source = get_tokenizer(config, train_dataset_raw, 'text')
    tokenizer_target = get_tokenizer(config, train_dataset_raw, 'label')

    # Convert to custom dataset
    train_dataset = TwitterDataset(train_dataset_raw, tokenizer_source, tokenizer_target, config['seq_len'])
    validation_dataset = TwitterDataset(validation_dataset_raw, tokenizer_source, tokenizer_target, config['seq_len'])

    max_len_source = 0
    max_len_target = 0

    for item in train_dataset_raw:
        source_ids = tokenizer_source.encode(item['text']).ids
        target_ids = tokenizer_source.encode(str(item['label'])).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))
    
    for item in validation_dataset_raw:
        source_ids = tokenizer_source.encode(item['text']).ids
        target_ids = tokenizer_source.encode(str(item['label'])).ids
        max_len_source = max(max_len_source, len(source_ids))
        max_len_target = max(max_len_target, len(target_ids))
    
    print(f'Max length of source sentence: {max_len_source}')
    print(f'Max length of target sentence: {max_len_target}')

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = 1, shuffle = True)

    return train_dataloader, validation_dataloader, tokenizer_source, tokenizer_target

def get_model(config, vocab_source_len, vocab_target_len):
    model = build_transformer(vocab_source_len, vocab_target_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def greedy_decode(model, source, source_mask, tokenizer_source, tokenizer_target, max_len, device):
    sos_idx = tokenizer_target.token_to_id('[SOS]')
    eos_idx = tokenizer_target.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token that is obtained from the decoder
    encoder_output = model.encode(source, source_mask)

    # Iteratively compute the output where each iteration outputs a single token in the predicted output
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Compute decoder output
        output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Obtain next token
        probabilities = model.project(output[:, -1])
        _, next_word = torch.max(probabilities, dim = 1)

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim = 1)

        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_dataset, tokenizer_source, tokenizer_target, max_len, device, print_message, global_state, writer, num_examples = 2):
    model.eval()
    count = 0

    # Size of the control window (default)
    console_width = 80
    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_source, tokenizer_target, max_len, device)
            source_text = batch['source_text'][0]
            target_text = batch['target_text'][0]
            model_output_text = tokenizer_target.decode(model_output.detach().cpu().numpy())

            print_message('-' * console_width)
            print_message(f'SOURCE: {source_text}')
            print_message(f'TARGET: {target_text}')
            print_message(f'PREDICTED: {model_output_text}')

            if count == num_examples:
                break

def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    train_dataloader, validation_dataloader, tokenizer_source, tokenizer_target = get_dataset(config)
    model = get_model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_source.token_to_id('[PAD]'), label_smoothing = 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            model.train()

            encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)
            projected_output = model.project(decoder_output) # (b, seq_len, target_vocab_size)

            label = batch['label'].to(device) # (batch, seq_len)

            # (batch, seq_len, target_vocab_size) -> (batch * seq_len, target_vocab_size)
            loss = loss_fn(projected_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            # Compute log loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropogation
            loss.backward()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        # Validate model
        run_validation(model, validation_dataloader, tokenizer_source, tokenizer_target, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # Save the model after every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
