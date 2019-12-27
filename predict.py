# encoding=utf-8

import torch

from TextSentiment import TextSentiment
import text_load

from torchtext.data.utils import ngrams_iterator


NGRAMS = 3

ag_news_label = {0 : "医学类",
                 1 : "非医学类"}

def tokenizer(text):
    return [word for word in text]

def predict(text, model, vocab, ngrams):

    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()


train_dataset, test_dataset = text_load._setup_datasets( ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N_EPOCHS = 20
min_valid_loss = float('inf')
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
vocab = train_dataset.get_vocab()
model.load_state_dict(torch.load('model/mode19.pt'))
# model = TextSentiment.to(device)

while(True) :
    ex_text_str =input("输入:")
    print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])


