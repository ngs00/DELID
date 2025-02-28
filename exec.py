import numpy
import random
from pandas import DataFrame
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader
from util.chem import load_elem_attrs
from util.data import load_calc_dataset, load_dataset, collate
from ml.gnn import *
from ml.delid import DELID


random_seed = 1
numpy.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)


dataset_name = 'lipop'
n_folds = 5
batch_size = 64
init_lr = 5e-4
l2_reg_coeff = 5e-6
n_epochs = 500
dim_latent = 64
dim_hidden = 64
list_r2 = list()
list_mae = list()


if dataset_name in ['esol', 'igc50', 'lc50']:
    dim_latent = 32
else:
    dim_latent = 64


elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
idx_calc_feats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
dataset_calc = load_calc_dataset(path_dataset='res/qm9_max6.xlsx',
                                 elem_attrs=elem_attrs,
                                 idx_smiles=0,
                                 idx_calc_feats=idx_calc_feats)
dataset = load_dataset(path_dataset='dataset/{}.xlsx'.format(dataset_name),
                       elem_attrs=elem_attrs,
                       idx_smiles=0,
                       idx_target=1,
                       dataset_calc=dataset_calc,
                       normalize_y=True)
k_folds = dataset.get_k_folds(n_folds=n_folds, random_seed=random_seed)


for k in range(0, n_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)

    gnn_frag = GIN(dim_node_feat=len(idx_calc_feats),
                   dim_hidden=dim_hidden,
                   dim_out=dim_hidden)
    model = DELID(dim_node_feat=dataset_train.data[0].mg.x.shape[1],
                  dim_edge_feat=dataset_train.data[0].mg.edge_attr.shape[1],
                  dim_latent=dim_latent,
                  dim_hidden=dim_hidden,
                  gnn_frag=gnn_frag).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=l2_reg_coeff)

    for epoch in range(0, n_epochs):
        loss_train = model.fit(loader_train, optimizer)
        print('Fold [{}/{}]\tEpoch [{}/{}]\tTrain loss: {:.3f}'.format(k + 1, n_folds, epoch + 1, n_epochs, loss_train))

    preds_test = model.predict(dataset_test, dataset_train.y_mean, dataset_train.y_std).numpy()
    targets_test = dataset_test.get_original_y().numpy()
    r2_test = r2_score(targets_test, preds_test)
    mae_test = mean_absolute_error(targets_test, preds_test)
    torch.save(model.state_dict(), 'save/model_{}_{}.pt'.format(dataset_name, k))

    list_r2.append(r2_test)
    list_mae.append(mae_test)
    pred_results = numpy.hstack([targets_test.reshape(-1, 1), preds_test.reshape(-1, 1)])
    DataFrame(pred_results).to_excel('save/preds_{}_{}.xlsx'.format(dataset_name, k), index=False, header=False)


print('Test R2-score: {:.3f} \u00B1 ({:.3f})'.format(numpy.mean(list_r2), numpy.std(list_r2)))
print('Test MAE: {:.3f} \u00B1 ({:.3f})'.format(numpy.mean(list_mae), numpy.std(list_mae)))
