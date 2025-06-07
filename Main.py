import torch
import torch.optim as optim
import numpy as np
import pickle as pickle
import scipy.sparse as ssp
import os.path
import random
import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from VGCN import *
import argparse
import psutil
import time
from util_functions import *
from model import *
from torch_geometric.loader import DataLoader


parser = argparse.ArgumentParser(description='Link Prediction')
# general settings
parser.add_argument('--data-name', default='BUP', help='network name')
parser.add_argument('--train-posname', default=None, help='train pos name')
parser.add_argument('--test-posname', default=None, help='test pos name')
parser.add_argument('--val-posname', default=None, help='val pos name')
parser.add_argument('--train-negname', default=None, help='train neg name')
parser.add_argument('--test-negname', default=None, help='test neg name')
parser.add_argument('--val-negname', default=None, help='test neg name')
parser.add_argument('--expression', default=None, help='Expression name')
parser.add_argument('--HV-num', default=None, help='high variable gene')
parser.add_argument('--celltype', default=None, help='cell type')
parser.add_argument('--net', default='Specific', help='netwokr type')
parser.add_argument('--fold', type=bool, default=False, help='5-fold validation')
parser.add_argument('--noise', type=float, default=False, help='Expression noise level')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
# model settings
parser.add_argument('--hop', default=2, metavar='S', help='enclosing subgraph hop number, options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=100, help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--batch-size', type=int, default=50, help='Batch size for training (default: 50)')
parser.add_argument('--alpha', type=float, default=1, help='Weight for loss component alpha (default: 1)')
parser.add_argument('--beta', type=float, default=1, help='Weight for loss component beta (default: 1)')
parser.add_argument('--hidden', type=int, default=100, help='Hidden layer size (default: 128)')
parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate (default: 5e-4)')
parser.add_argument('--d-model', type=int, default=100, help='Transformer model dimension (default: 100)')
parser.add_argument('--nhead', type=int, default=2, help='Number of attention heads (default: 2)')
parser.add_argument('--neg', type=int, default=6, help='positive negative ratio')
parser.add_argument('--layer', type=int, default=2, help='GCN layers num')
parser.add_argument('--window-size', type=int, default=50, help='Window Size')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

random.seed(args.seed)
np.random.seed(args.seed) 
torch.manual_seed(args.seed)

if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)



def loop_dataset_gem(classifier, loader, optimizer=None):
    total_loss = []
    all_targets = []
    all_scores = []
    latent_emb = []
    
    pbar_unit = 'batch' #if optimizer else 'iteration'
    pbar = tqdm(loader, unit=pbar_unit)

    n_samples = 0
    # predictions = []
    # feat = []

    for batch in pbar:    
        all_targets.extend(batch.y.tolist())
        logits, loss, acc, loss1, loss2, loss3, latent = classifier(batch)
        all_scores.append(logits.cpu().detach())
        latent_emb.append(latent.cpu().detach())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        
        pbar.set_description('Loss_e: %0.5f Loss_l: %0.5f Loss_c: %0.5f' % (loss1, loss2, loss3) )
        total_loss.append( np.array([loss, acc]) * len(batch.y))
        
        n_samples += len(batch.y)

    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()
    # latent_emb = torch.cat(latent_emb).cpu().numpy()

    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    # latent_emb_df = pd.DataFrame(latent_emb)
    # latent_emb_df.to_csv('latent_emb.csv', index=False)

    all_targets = np.array(all_targets)
    avg_precision = average_precision_score(all_targets, all_scores)
    fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    avg_loss = np.concatenate((avg_loss, [auc, avg_precision])) # type: ignore
    
    return avg_loss



def train_test(sample):
    '''数据准备'''
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))
    args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))
    
    # Import training positive samples, testing positive samples
    if args.fold:
        args.train_dir = os.path.join(args.file_dir, 'Data_5fold/{}/{} {}/{}/{}_{}_train_pos.txt'.format(args.net, args.celltype, args.HV_num, sample, args.celltype, args.HV_num))
        args.test_dir = os.path.join(args.file_dir, 'Data_5fold/{}/{} {}/{}/{}_{}_test_pos.txt'.format(args.net, args.celltype, args.HV_num, sample, args.celltype, args.HV_num))
        args.val_dir = os.path.join(args.file_dir, 'Data_5fold/{}/{} {}/{}/{}_{}_val_pos.txt'.format(args.net, args.celltype, args.HV_num, sample, args.celltype, args.HV_num))
    else:
        args.train_dir = os.path.join(args.file_dir, 'Data/{}/{}/TFs+{}/{}_{}_train_pos.txt'.format(args.net, args.celltype, args.HV_num, args.celltype, args.HV_num))
        args.test_dir = os.path.join(args.file_dir, 'Data/{}/{}/TFs+{}/{}_{}_test_pos.txt'.format(args.net, args.celltype, args.HV_num, args.celltype, args.HV_num))
        args.val_dir = os.path.join(args.file_dir, 'Data/{}/{}/TFs+{}/{}_{}_val_pos.txt'.format(args.net, args.celltype, args.HV_num, args.celltype, args.HV_num))
    train_pos_idx = np.loadtxt(args.train_dir, dtype=int)
    test_pos_idx = np.loadtxt(args.test_dir, dtype=int)
    val_pos_idx = np.loadtxt(args.val_dir, dtype=int)
    # Build training negative samples, testing negative samples
    if args.fold:
        args.train_dir = os.path.join(args.file_dir, 'Data_5fold/{}/{} {}/{}/{}_{}_train_neg.txt'.format(args.net, args.celltype, args.HV_num, sample, args.celltype, args.HV_num))
        args.test_dir = os.path.join(args.file_dir, 'Data_5fold/{}/{} {}/{}/{}_{}_test_neg.txt'.format(args.net, args.celltype, args.HV_num, sample, args.celltype, args.HV_num))
        args.val_dir = os.path.join(args.file_dir, 'Data_5fold/{}/{} {}/{}/{}_{}_val_neg.txt'.format(args.net, args.celltype, args.HV_num, sample, args.celltype, args.HV_num))
    else:
        args.train_dir = os.path.join(args.file_dir, 'Data/{}/{}/TFs+{}/{}_{}_train_neg.txt'.format(args.net, args.celltype, args.HV_num, args.celltype, args.HV_num))
        args.test_dir = os.path.join(args.file_dir, 'Data/{}/{}/TFs+{}/{}_{}_test_neg.txt'.format(args.net, args.celltype, args.HV_num, args.celltype, args.HV_num))
        args.val_dir = os.path.join(args.file_dir, 'Data/{}/{}/TFs+{}/{}_{}_val_neg.txt'.format(args.net, args.celltype, args.HV_num, args.celltype, args.HV_num))
    train_neg_idx = np.loadtxt(args.train_dir, dtype=int)
    test_neg_idx = np.loadtxt(args.test_dir, dtype=int)
    val_neg_idx = np.loadtxt(args.val_dir, dtype=int)

    max_idx = max(np.max(train_pos_idx), np.max(test_pos_idx), np.max(val_pos_idx), np.max(train_neg_idx), np.max(test_neg_idx), np.max(val_neg_idx))
    net = ssp.csc_matrix((np.ones(len(train_pos_idx)), (train_pos_idx[:, 0], train_pos_idx[:, 1])), shape=(max_idx+1, max_idx+1))
    net[train_pos_idx[:, 1], train_pos_idx[:, 0]] = 1 
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  

    train_pos = (train_pos_idx[:, 0], train_pos_idx[:, 1])   
    test_pos = (test_pos_idx[:, 0], test_pos_idx[:, 1])      
    val_pos = (val_pos_idx[:, 0], val_pos_idx[:, 1])
    train_neg = (train_neg_idx[:, 0], train_neg_idx[:, 1])   
    test_neg = (test_neg_idx[:, 0], test_neg_idx[:, 1])      
    val_neg = (val_neg_idx[:, 0], val_neg_idx[:, 1])
    
    
    '''Train and apply classifier'''
    A = net.copy()  # the observed network
    A.eliminate_zeros()
    print(A.shape)
    
    
    '''Import gene expression'''
    path = os.path.join(args.file_dir, 'Expression/{}/{}/TFs+{}/BL--ExpressionData.csv'.format(args.net, args.celltype, args.HV_num))
    information = pd.read_csv(path)
    information = information.iloc[:, 1:].to_numpy()
    scaler = StandardScaler()       
    information = scaler.fit_transform(information)
    
    
    '''Dimensionality reduction using VAE'''
    x = torch.tensor(information, dtype=torch.float32)
    model = VAE(information.shape[1], 128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    min_loss = float('inf')
    best_model_state_dict = None
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()  
        optimizer.zero_grad()  
        reconstruction, mu, logvar = model(x)
        loss = model.compute_loss(x, reconstruction, mu, logvar)

        loss.backward()
        optimizer.step()
        avg_loss = loss.item()  
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_model_state_dict = model.state_dict()

    best_model_path = f'best_model/VAE/{args.net}/{args.celltype}_{args.HV_num}_best_model.pth'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    torch.save(best_model_state_dict, best_model_path)
    # print(f"Best model saved with loss: {min_loss:.4f}")
    model.load_state_dict(best_model_state_dict)
    model.eval()  
    with torch.no_grad(): 
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)  
    node_information = z.cpu().detach().numpy()

    
    '''Subgraph Extraction'''
    window_size = args.window_size
    args.d_model = args.window_size * 2
    if args.net == 'Specific':
        flag = 1
    else:
        flag = None

    train_graphs, test_graphs, val_graphs = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, val_pos, val_neg, args.hop, args.max_nodes_per_hop, node_information, window_size, flag)
    print(('# train: %d, # test: %d, # val: %d' % (len(train_graphs), len(test_graphs), len(val_graphs))))

    
    '''Line graph transformation'''
    train_lines = to_linegraphs(train_graphs)      
    test_lines = to_linegraphs(test_graphs)
    val_lines = to_linegraphs(val_graphs)
    
    
    '''Initialize model'''
    if args.layer==2:
        args.latent_dim2 = [128, 128]
    else:
        if args.layer==3:
            args.latent_dim2 = [128, 128, 128]
        else:
            if args.layer==4:
                args.latent_dim2 = [128, 128, 128, 128]
            else:
                args.latent_dim2 = [128]

    args.dropout = 0.5
    args.mode = 'gpu'
    args.num_epochs = 50
    args.feat_dim1 = window_size*2
    args.feat_dim2 = node_information.shape[1]*2
    
    train_loader = DataLoader(train_lines, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_lines, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_lines, batch_size=args.batch_size, shuffle=False)
    
    
    classifier = Net(args.feat_dim1, args.feat_dim2, args.hidden, args.latent_dim2, 
                            args.alpha, args.beta, args.nhead, args.d_model, args.dropout)
    if args.mode == 'gpu':
            classifier = classifier.to("cuda")
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    
    
    '''Model running'''
    val_best = [0, 0, 0, 0]
    counter = 0
    patience = 10
    best_model_path = f'best_model/model/{args.net}/{args.celltype}_{args.HV_num}_best_model.pth'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    for epoch in range(args.num_epochs):
        # Model training
        classifier.train()
        avg_loss = loop_dataset_gem(classifier, train_loader, optimizer=optimizer)
        print(('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3])))
    
        # Model evaluation
        classifier.eval()
        val_loss = loop_dataset_gem(classifier, val_loader, None)
        print(('\033[31maverage val of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (epoch, val_loss[0], val_loss[1], val_loss[2], val_loss[3])))
    
        # Early stopping
        if val_best[2] < val_loss[2]:
            val_best[0] = val_loss[0]
            val_best[2] = val_loss[2]
            val_best[3] = val_loss[3]
            torch.save(classifier.state_dict(), best_model_path)  # Save model
            print(f'Best model saved at epoch {epoch} with AUC: {val_best[2]:.5f}')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered after {epoch} epochs.')
                break  
    
    # Evaulation model
    classifier.eval()
      
    # Load the best model
    classifier.load_state_dict(torch.load(best_model_path))
    print('Best model loaded for testing.')
    with torch.no_grad():
        test_results = loop_dataset_gem(classifier, test_loader, None)
    
    print('\033[95mTest performance: loss %.5f acc %.5f auc %.5f ap %.5f\033[0m' % (test_results[0], test_results[1], test_results[2], test_results[3]))

    # Record the results
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('RECORD.txt', 'a') as f:
        f.write('[{}] {} {} {} {}: auc {:.5f} ap {:.5f} loss {:.5f}\n'.format(
            current_time, args.net, args.celltype, args.HV_num, sample, test_results[2], test_results[3], test_results[0]
        ))

    return test_results, val_best
    

if __name__ == '__main__':
    sample_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']
    fold_list = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    
    # Measure memory usage
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # in MB
    # Measure time
    start_time = time.time()

    print(args.fold)
    if args.fold:
        # for sample in sample_list:
        #     test_results, val_results = train_test(sample)
        for fold in fold_list:
            test_results, val_results = train_test(fold)
    else:
        test_results, val_results = train_test(None)
    # torch.cuda.empty_cache()
    end_time = time.time()
    memory_after = process.memory_info().rss / 1024 / 1024  # in MB

    # Print results
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Memory usage before: {memory_before:.2f} MB")
    print(f"Memory usage after: {memory_after:.2f} MB")
    print(f"Memory usage difference: {memory_after - memory_before:.2f} MB")
