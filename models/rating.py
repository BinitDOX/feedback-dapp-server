import torch
from torch import nn
from torch import optim
from transformers import BertTokenizer
from transformers import BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  # Get pretrained tokenizer 

class BertClassifier(nn.Module):

    def __init__(self, n_hidden=768, dropout=0.3):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(n_hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask, verbose=False):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_output = self.relu(linear_output)
        
        if verbose == True:
            print('InputID: ', input_id.shape)
            print('Mask: ', mask.shape)
            print('Pool: ', pooled_output.shape)
            print('Drop: ', dropout_output.shape)
            print('Linear: ', linear_output.shape)
            print('Final: ', final_output.shape)
        
        return linear_output
        
        
        
def load_model(model, path):
    print('Load model:', path)
    model_stats = torch.load(path, map_location=torch.device(conf.device))
    model.load_state_dict(model_stats['net_param'])
    return model
    
    
class Configuration():  # General configuration class
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU or CPU
        self.n_epochs = 3  # Number of full iterations on data
        self.tokenizer_maxlen = 64
        self.lr = 1e-6  # Learning rate
        self.s_batch = 16  # Batch size (train), lower this if out of memory
        self.valid_freq = 1  # Epoch based
        self.printloss_freq = 500  # Batch based
        self.plotloss_freq = 500  # Batch based
        self.save_freq = 1  # Epoch based
        self.checkpoint_path = ''  # Fine tuned model path
        
conf = Configuration()


model = BertClassifier().to(conf.device)


model = load_model(model, 'models/rating.ckpt-3')  # Load model

def get_rating(feedback, verbose=False):
    model.eval()
    with torch.no_grad():
        tokenized_fb = tokenizer(str(feedback), padding='max_length', max_length=conf.tokenizer_maxlen, truncation=True,
                return_tensors="pt")

        att_mask = tokenized_fb['attention_mask'].to(conf.device)
        input_id = tokenized_fb['input_ids'].squeeze(1).to(conf.device)

        output = model(input_id, att_mask)
        
        if verbose:
            print(output, output.shape)
    model.train()

    # Normalize ?:
    return output.item()
    
    
def get_ratings(feedbacks):
    ratings = []
    for f in feedbacks:
        ratings.append(get_rating(f))
    
    minr = min(ratings)
    maxr = max(ratings)
    
    norm_ratings = []
    for r in ratings:
        norm_ratings.append((r - minr) / (maxr - minr)) * 100
        
    return str(sum(norm_ratings)/len(norm_ratings))