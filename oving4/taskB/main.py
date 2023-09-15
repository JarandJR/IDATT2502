import torch
import numpy as np

from model import LongShortTermMemoryModel

emojis = {
    'hat': '🎩',
    'rat': '🐭',
    'cat': '😺',
    'flat': '🏢',
    'matt': '🙋',
    'cap': '🧢',
    'son': '👦'
}

emoji_aliases = {
    '🎩': ':hat:',
    '🐭': ':rat:',
    '😺': ':cat:',
    '🙋': ':matt:',
    '🧢': ':cap:',
    '👦': ':son:',
    '🏢': ':flat:'
}

index_to_emoji =[value for _,value in emojis.items()]

index_to_char = [' ', 'h', 'a', 't', 'r','c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

char_encodings = np.eye(len(index_to_char))
encoding_size = len(char_encodings)
emojies = np.eye(len(emojis))
emoji_encoding_size=len(emojies)

letters ={}

for i, letter in enumerate(index_to_char):
        letters[letter] = char_encodings[i]


x_train = torch.tensor([
        [[letters['h']], [letters['a']], [letters['t']], [letters[' ']]],
        [[letters['r']], [letters['a']], [letters['t']], [letters[' ']]],
        [[letters['c']], [letters['a']], [letters['t']], [letters[' ']]],
        [[letters['f']], [letters['l']], [letters['a']], [letters['t']]],
        [[letters['m']], [letters['a']], [letters['t']], [letters['t']]],
        [[letters['c']], [letters['a']], [letters['p']], [letters[' ']]],
        [[letters['s']], [letters['o']], [letters['n']], [letters[' ']]],
        ], 
        dtype=torch.float)

y_train = torch.tensor([
        [emojies[0], emojies[0], emojies[0], emojies[0]] ,
        [emojies[1], emojies[1], emojies[1], emojies[1]],
        [emojies[2], emojies[2], emojies[2], emojies[2]],
        [emojies[3], emojies[3], emojies[3], emojies[3]],
        [emojies[4], emojies[4], emojies[4], emojies[4]],
        [emojies[5], emojies[5], emojies[5], emojies[5]],
        [emojies[6], emojies[6], emojies[6], emojies[6]]], 
        dtype=torch.float)

model = LongShortTermMemoryModel(encoding_size=encoding_size, emoji_encoding_size= emoji_encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
print("\ntraining..\n")
optimizer = torch.optim.RMSprop(model.parameters(), 0.001)  # 0.001
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()

def generate_emoji(string):
    y = -1
    model.reset()
    for i in range(len(string)):
        char_index = index_to_char.index(string[i])
        y = model.f(torch.tensor([[char_encodings[char_index]]], dtype=torch.float))
    emoji = index_to_emoji[y.argmax(1)]
    print(emoji)

    # If you can't print emojis
    #emoji_alias = emoji_aliases.get(emoji, emoji)
    #print(emoji_alias)

generate_emoji('rt')
generate_emoji('rats')
generate_emoji('hat')
generate_emoji('fat')