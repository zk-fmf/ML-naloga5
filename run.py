import pickle
from train import Training
from test import Testing

if __name__ == '__main__': 

    main_path_to_data = '/data/PSUF_naloge/5-naloga/processed' 
    unique_name_for_this_run = 'yolo123' # neobvezno  

    # Nalozimo sezname za ucno, validacijsko in testno mnozico
    with open (main_path_to_data + "/train_info", 'rb') as fp:
        train_info = pickle.load(fp)
    with open (main_path_to_data + "/val_info", 'rb') as fp:
        valid_info = pickle.load(fp)
    with open (main_path_to_data + "/test_info", 'rb') as fp:
        test_info = pickle.load(fp) 

    # Nastavimo hiperparametre v slovarju
    hyperparameters = {}
    hyperparameters['learning_rate'] = 0.2e-3 # learning rate
    hyperparameters['weight_decay'] = 0.0001 # weight decay
    hyperparameters['total_epoch'] = 20 # total number of epochs
    hyperparameters['multiplicator'] = 0.95 # each epoch learning rate is decreased on LR*multiplicator

    # Ustvarimo ucni in testni razred
    TrainClass = Training(main_path_to_data, unique_name=unique_name_for_this_run)
    TestClass = Testing(main_path_to_data)

    # Naucimo model za izbrane hiperparametre
    aucs, losses, path_to_model = TrainClass.train(train_info, valid_info, hyperparameters)

    # Najboljsi model glede na validacijsko mnozico (zadnji je /LAST_model.pth)
    best_model = path_to_model + '/BEST_model.pth' 

    # Testiramo nas model na testni mnozici
    auc, fpr, tpr, thresholds, trues, predictions = TestClass.test(test_info, best_model)

    print("Test set AUC result: ", auc)