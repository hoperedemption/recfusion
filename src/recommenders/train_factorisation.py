import pandas as pd 
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import factorisation

# PyTorch tensorboard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# read the dataset from the CSV file
book_df = pd.read_csv('../datasets/book_reviews/archive/Ratings.csv')

# find out the number of users using User-ID
first_user_id = book_df['User-ID'].min()
last_user_id = book_df['User-ID'].max()
num_users = last_user_id - first_user_id + 1

# find out the number of items using ISBN
first_item_id = book_df['ISBN'].min()
last_item_id = book_df['ISBN'].max()
num_items = last_item_id - first_item_id + 1

# filter out users without ratings
filtered_df = book_df[book_df['Book-Rating'] >= 1]

# replace 0 ratings with NaNs
book_df['Book-Rating'] = book_df['Book-Rating'].replace(0, np.nan)

# filtered out group by user_id
train, test = train_test_split(filtered_df, test_size=0.2, random_state=42)

# convert to PyTorch
user_train = torch.tensor(train['User-ID'].values)
item_train = torch.tensor(train['ISBN'].values)
rating_train = torch.tensor(train['Book-Rating'].values)

user_test = torch.tensor(test['User-ID'].values)
item_test = torch.tensor(test['ISBN'].values)
rating_test = torch.tensor(test['Book-Rating'].values)

# create the datasets
train_dataset = torch.utils.data.TensorDataset(user_train, item_train, rating_train)
test_dataset = torch.utils.data.TensorDataset(user_test, item_test, rating_test)

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# hyperparameter for factorisation
latent_factor = 10

# define the model
model = factorisation.MatrixFactorization(num_users, num_items, latent_factor)

# define the optimizer (took default PyTorch one)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# define the regularization term (lambda)
λ = 0.001

# define the loss function
def loss_function_mse(user, item, output, target, λ):
    return (output - target).pow(2) + λ * (model.P(user).pow(2).sum() + model.Q(item).pow(2).sum() + model.b_user(user).pow(2) + model.b_item(item).pow(2))

# define the rmse function
def rmse(prediction, target):
    return torch.sqrt((prediction - target).pow(2))

# define the training for one epoch
def train_one_epoch(epoch_index, tb_writer, training_loader):
    running_loss = 0.0
    last_loss = 0.0
    counter = 0
    total_loss = 0

    for i, data in enumerate(training_loader):

        # unpack the data
        user, item, rating = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # compute the forward pass
        output = model(user, item)
        # compute the loss and its gradients
        loss = loss_function_mse(user, item, output, rating, λ)
        loss.backward()
        # update the parameters
        optimizer.step()
        # update the running loss
        running_loss += loss.item()

        # save info
        total_loss += running_loss
        counter = i + 1

        # print the progress
        if i % 100 == 99:
            last_loss = running_loss / 100
            print(f'Epoch {epoch_index+1}, Batch {i+1}/{len(training_loader)}, Loss: {last_loss:.4f}')
            tb_writer.add_scalar('Train Loss', last_loss / 100, epoch_index * len(training_loader) + i + 1)
            running_loss = 0.0

    return total_loss / counter

# define the training loop
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tb_writer = SummaryWriter(f'runs/factorisation_{timestamp}')
epoch_number = 0

EPOCHS = 10

best_v_loss = 1_000_000_000

for epoch_nb in range(EPOCHS):
    print(f'Epoch {epoch_nb + 1}')

    # make sure gradient tracking is enabled
    model.train(True)
    avg_train_loss = train_one_epoch(epoch_nb, tb_writer, train_dataloader)

    running_loss = 0.0
    # set the model to evaluation mode
    model.eval(True)
    
    # disable gradient tracking
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            user, item, rating = data
            output = model(user, item)
            loss = rmse(output, rating)
            running_loss += loss.item()

        avg_test_loss = running_loss / len(test_dataloader)
        print(f'Test Loss: {avg_test_loss:.4f}')

        # save the model if it's the best
        if avg_test_loss < best_v_loss:
            best_v_loss = avg_test_loss
            torch.save(model.state_dict(), f'models/factorisation_{timestamp}.pth')
            print('Model saved!')
    

