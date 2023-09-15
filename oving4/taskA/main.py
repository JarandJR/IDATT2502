import torch

from model import LongShortTermMemoryModel

char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0],  # 'h'
    [0., 0., 1., 0., 0., 0., 0., 0],  # 'e'
    [0., 0., 0., 1., 0., 0., 0., 0],  # 'l'
    [0., 0., 0., 0., 1., 0., 0., 0],  # 'o'
    [0., 0., 0., 0., 0., 1., 0., 0],  # 'w'
    [0., 0., 0., 0., 0., 0., 1., 0],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 1],  # 'd'
]
encoding_size = len(char_encodings)

index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

x_train = torch.tensor([[char_encodings[0]], [char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]],
                        [char_encodings[4]], [char_encodings[0]], [char_encodings[5]], [char_encodings[4]], [char_encodings[6]],
                        [char_encodings[3]], [char_encodings[7]], [char_encodings[0]]])  # ' hello world '
y_train = torch.tensor([char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4],
                        char_encodings[0], char_encodings[0], char_encodings[5], char_encodings[4], char_encodings[6],
                        char_encodings[3], char_encodings[7], char_encodings[0]])  # 'hello world '

model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
print("training..")
for epoch in range(500):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        # Generate characters from the initial characters ' h'
        model.reset()
        text = ' h'
        model.f(torch.tensor([[char_encodings[0]]]))
        y = model.f(torch.tensor([[char_encodings[1]]]))
        text += index_to_char[y.argmax(1)]
        for c in range(50):
            y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]]))
            text += index_to_char[y.argmax(1)]
        print(text)
