import numpy as np
import torch
from torch.utils.data import DataLoader
from model import ModelCT
from datareader import DataReader
from sklearn.metrics import roc_auc_score
import time
import os
import datetime

class Training():
    def __init__(self, main_path_to_data, unique_name='mytestrun123'):
        self.main_path_to_data = main_path_to_data
        self.unique_name = unique_name

    def train(self, train_info, valid_info, hyperparameters):
        """ Train the model. """
        
        # 1. Create folders to save the model
        timedate_info= str(datetime.datetime.now()).split(' ')[0] + '_' + str(datetime.datetime.now().strftime("%H:%M:%S")).replace(':', '-')
        path_to_model = 'trained_models/' + self.unique_name +  '_' + timedate_info
        os.mkdir(path_to_model)
        
        # 2. Load hyperparameters
        learning_rate = hyperparameters['learning_rate']
        weight_decay = hyperparameters['weight_decay']
        total_epoch = hyperparameters['total_epoch']
        multiplicator = hyperparameters['multiplicator']
        
        # 3. Consider class imbalance
        negative, positive = 0, 0
        for _, label in train_info:
            if label == 0:
                negative += 1
            elif label == 1:
                positive += 1
        
        pos_weight = torch.Tensor([(negative/positive)]).cpu()
        
        # 4. Create train and validation generators, batch_size = 10 for validation generator (10 central slices)
        train_datareader = DataReader(self.main_path_to_data, train_info)
        train_generator = DataLoader(train_datareader, batch_size=16, shuffle=True, pin_memory=True, num_workers=2)
        
        valid_datareader = DataReader(self.main_path_to_data, valid_info)
        valid_generator = DataLoader(valid_datareader, batch_size=10, shuffle=False, pin_memory=True, num_workers=2)
        
        # 5. Prepare model
        Model = ModelCT()
        Model.cpu()
        
        # 6. Define criterion function, optimizer and scheduler
        citerion_clf = torch.nn.BCEWithLogitsLoss(pos_weight) # pos_weight for class imbalance
        optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, multiplicator, last_epoch=-1)
        
        # 7. Creat lists for tracking AUC and Losses during training
        aucs = []
        losses = []
        best_auc = -np.inf
        nb_batches = len(train_generator)
        
        # 8. Run training
        for epoch in range(total_epoch):
            start = time.time()
            print('Epoch: %d/%d' % (epoch + 1, total_epoch))
            running_loss = 0
            
            # A) Train model
            Model.train()  # put model in training mode
            for item_train in train_generator:  
                # Forward pass          
                optimizer.zero_grad()
                loss = Model.train_update(item_train, citerion_clf)
                # Backward pass
                loss.backward()
                optimizer.step()
                # Track loss change
                running_loss += loss.item()
            
            # B) Validate model
            predictions = []
            trues = []
            
            Model.eval() # put model in eval mode
            for item_valid in valid_generator:
                prediction = Model.predict(item_valid, is_prob=True)
                predictions.append(np.mean(prediction.cpu().numpy()))
                trues.append(item_valid[1].numpy()[0])
        
            auc = roc_auc_score(trues, predictions)
            
            # C) Track changes, update LR, save best model
            print("AUC: ", auc, ", Running loss: ", running_loss/nb_batches, ", Time: ", time.time()-start)
            
            if (epoch >= total_epoch//2) and (auc > best_auc): # If over 1/2 of epochs and best AUC, save model as best model.
                torch.save(Model.state_dict(), path_to_model + '/BEST_model.pth')
                best_auc = auc         
            else:
                pass
            
            aucs.append(auc)
            losses.append(running_loss/nb_batches)
            scheduler.step()
            
        np.save(path_to_model + '/AUCS.npy', np.array(aucs))
        np.save(path_to_model + '/LOSSES.npy', np.array(losses))
        torch.save(Model.state_dict(), path_to_model + '/LAST_model.pth')
        
        return aucs, losses, path_to_model