import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    
    sns.set_theme()
    
    suffix_list = [
                   'APA', 'CVX', 'MRO', 'XOM'
                                                        ]
    
    for suffix in suffix_list:
        print(suffix)

        sa, sc, la, lc = [], [], [], []
        for i in range(1,3+1):
    #         print('loading training loss...')
    #         sa_train = np.load(f'./results_{suffix}/sa/train_loss.npy', allow_pickle=True)
    #         sc_train = np.load(f'./results_{suffix}/sc/train_loss.npy', allow_pickle=True)
    #         la_train = np.load(f'./results_{suffix}/la/train_loss.npy', allow_pickle=True)
    #         lc_train = np.load(f'./results_{suffix}/lc/train_loss.npy', allow_pickle=True)
                  
            print('loading validation loss...')
            sa_val = np.load(f'./results_{i}/results_{suffix}/sa/val_loss.npy', allow_pickle=True)
            sc_val = np.load(f'./results_{i}/results_{suffix}/sc/val_loss.npy', allow_pickle=True)
            la_val = np.load(f'./results_{i}/results_{suffix}/la/val_loss.npy', allow_pickle=True)
            lc_val = np.load(f'./results_{i}/results_{suffix}/lc/val_loss.npy', allow_pickle=True)

            sa.append(sa_val)
            sc.append(sc_val)
            la.append(la_val)
            lc.append(lc_val)


        sa = np.array(sa)
        sc = np.array(sc)
        la = np.array(la)
        lc = np.array(lc)

        sa_mean = np.average(sa, axis=0)
        sc_mean = np.average(sc, axis=0)
        la_mean = np.average(la, axis=0)
        lc_mean = np.average(lc, axis=0)

        sa_std = np.std(sa, axis=0)
        sc_std = np.std(sc, axis=0)
        la_std = np.std(la, axis=0)
        lc_std = np.std(lc, axis=0)

        fig = plt.figure(figsize=(5,5))

        indices = np.arange(1,len(sa_val)+1)

#         plt.plot(indices, sa_train, label='sa-train')
#         plt.plot(indices, sc_train, label='sc-train')
#         plt.plot(indices, la_train, label='la-train')
#         plt.plot(indices, lc_train, label='lc-train')

        plt.plot(indices, sa_mean, label='sa-val')
        plt.plot(indices, sc_mean, label='sc-val')
        plt.plot(indices, lc_mean, label='lc-val')
        plt.plot(indices, la_mean, label='la-val')

        plt.fill_between(indices, sa_mean-sa_std, sa_mean+sa_std, alpha=0.2)
        plt.fill_between(indices, sc_mean-sc_std, sc_mean+sc_std, alpha=0.2)
        plt.fill_between(indices, la_mean-la_std, la_mean+la_std, alpha=0.2)
        plt.fill_between(indices, lc_mean-lc_std, lc_mean+lc_std, alpha=0.2)

        plt.title(f'{suffix}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        
        print(suffix)
        print('avg:', sa_mean[-1], sc_mean[-1], la_mean[-1], lc_mean[-1])
        print('std:', sa_std[-1], sc_std[-1], la_std[-1], lc_std[-1])
                

        # plt.show()
        plt.savefig(f'fig_{suffix}')
        
#         fig = plt.figure(figsize=(5,5))
#         plt.errorbar([ sa_mean[-1], sc_mean[-1], la_mean[-1], lc_mean[-1]], yerr=[sa_std[-1], sc_std[-1], la_std[-1], lc_std[-1]])
        
