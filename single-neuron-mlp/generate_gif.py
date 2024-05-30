



import os
import numpy as np
import matplotlib.pyplot as plt

import imageio



def generate_learning_rate_gif(X, y, learning_rate_list, weights_list, weights_init_list, min_values, max_values, folderpath=""):

    #plot_folderpath = 'plots/MLP_one_neuron_analysis-v2/model_predictions_lr/'
    os.makedirs(folderpath, exist_ok=True)

    print("images generation")
    fig, axs = plt.subplots(nrows=1,ncols=5, figsize=(10,2.5), sharey=True)
    x_range = np.linspace(min_values[0], max_values[0], 100)
    x_norm_range = np.linspace(0, 1, 100)
    for i in range(5):
        axs[i].scatter(X[:,0], y, color="blue", alpha=0.05)
        w0, w1 = weights_init_list[i][0][0], weights_init_list[i][1][0]
        axs[i].plot(x_range,  w1*x_norm_range + w0, color="red")
        axs[i].set_title(f"lr={learning_rate_list[i]}\n$w_0={np.round(w0,1)}$, $w_1={np.round(w1,1)}$")
        axs[i].set_xlabel("X")
        if i == 0:
            axs[i].set_ylabel("y")
    fig.suptitle(f'Model predictions - weights initialization', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{folderpath}plot_0.png')
    plt.close()
    for j in range(500):
        fig, axs = plt.subplots(nrows=1,ncols=5, figsize=(10,2.5), sharey=True)
        for i in range(5):
            axs[i].scatter(X[:,0], y, color="blue", alpha=0.05)
            w0, w1 = np.array(weights_list[i]).reshape(500,2)[j,0], np.array(weights_list[i]).reshape(500,2)[j,1]
            axs[i].plot(x_range,  w1*x_norm_range + w0, color="red")
            axs[i].set_title(f"lr={learning_rate_list[i]}\n$w_0={np.round(w0,1)}$, $w_1={np.round(w1,1)}$")
            axs[i].set_xlabel("X")
            if i == 0:
                axs[i].set_ylabel("y")
        fig.suptitle(f'Model predictions - epoch={j+1}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{folderpath}plot_{j+1}.png')
        plt.close()
        print(f"\r{j+1}/500", end="")
    print()
    
    images = []
    selected_epochs = sum([
        np.arange(0,15).astype(int).tolist(),
        np.arange(15,100,10).astype(int).tolist(),
        np.arange(100,510,10).astype(int).tolist()
    ],[])
    for i in selected_epochs:
        filename = f'{folderpath}plot_{i}.png'
        images.append(imageio.imread(filename))

    print("gif creation:", end="")
    imageio.mimsave(f'{folderpath}animated_plot.gif', images, duration=0.3, loop=0)

    print(" ok")



def generate_batch_size_gif(X, y, batch_size_list, weights_list, weights_init_list, min_values, max_values, folderpath=""):

    #plot_folderpath = 'plots/MLP_one_neuron_analysis-v2/model_predictions_bs/'
    os.makedirs(folderpath, exist_ok=True)

    print("images generation")
    fig, axs = plt.subplots(nrows=1,ncols=5, figsize=(10,2.5), sharey=True)
    x_range = np.linspace(min_values[0], max_values[0], 100)
    x_norm_range = np.linspace(0, 1, 100)
    for i in range(5):
        axs[i].scatter(X[:,0], y, color="blue", alpha=0.05)
        w0, w1 = weights_init_list[i][0][0], weights_init_list[i][1][0]
        axs[i].plot(x_range,  w1*x_norm_range + w0, color="red")
        axs[i].set_title(f"bs={batch_size_list[i]}\n$w_0={np.round(w0,1)}$, $w_1={np.round(w1,1)}$")
        axs[i].set_xlabel("X")
        if i == 0:
            axs[i].set_ylabel("y")
    fig.suptitle(f'Model predictions - weights initialization', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{folderpath}plot_0.png')
    plt.close()

    for j in range(500):
        fig, axs = plt.subplots(nrows=1,ncols=5, figsize=(10,2.5), sharey=True)
        for i in range(5):
            axs[i].scatter(X[:,0], y, color="blue", alpha=0.05)
            w0, w1 = np.array(weights_list[i]).reshape(500,2)[j,0], np.array(weights_list[i]).reshape(500,2)[j,1]
            axs[i].plot(x_range,  w1*x_norm_range + w0, color="red")
            axs[i].set_title(f"bs={batch_size_list[i]}\n$w_0={np.round(w0,1)}$, $w_1={np.round(w1,1)}$")
            axs[i].set_xlabel("X")
            if i == 0:
                axs[i].set_ylabel("y")
        fig.suptitle(f'Model predictions - epoch={j+1}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{folderpath}plot_{j+1}.png')
        plt.close()
        print(f"\r{j+1}/500", end="")
    print()

    images = []
    selected_epochs = sum([
        np.arange(0,15).astype(int).tolist(),
        np.arange(15,100,10).astype(int).tolist(),
        np.arange(100,510,10).astype(int).tolist()
    ],[])
    for i in selected_epochs:
        filename = f'{folderpath}plot_{i}.png'
        images.append(imageio.imread(filename))

    print("gif creation:", end="")
    imageio.mimsave(f'{folderpath}animated_plot.gif', images, duration=0.3, loop=0)
    print(" ok")



def generate_batch_size_steps_gif(X, y, i, batch_size_list, batch_indexes_list, batch_weights_list, weights_init_list, min_values, max_values, folderpath="", epochs=None):
    # i : index of the batch size in batch_size_list


    #plot_folderpath = f'plots/MLP_one_neuron_analysis-v2/batch_size_model_update_visualization/bs{batch_size_list[i]}/'
    os.makedirs(folderpath, exist_ok=True)

    print("images generation")
    x_range = np.linspace(min_values[0], max_values[0], 100)
    x_norm_range = np.linspace(0, 1, 100)

    plt.scatter(X[:,0], y, color="blue", alpha=0.05)
    w0, w1 = weights_init_list[i][0][0], weights_init_list[i][1][0]
    plt.plot(x_range,  w1*x_norm_range + w0, color="red")
    plt.title(f"Model predictions - batch_size={batch_size_list[i]}, initialization")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.savefig(f'{folderpath}plot_0.png')
    plt.close()

    previous_model = w1*x_norm_range + w0
    X_indexes = np.arange(X.shape[0])
    #n_batch = np.array(batch_indexes_list[i]).shape[1]
    n_batch = len(batch_indexes_list[i][0])

    cpt = 1
    #epochs = [0, 1, 2, 10, 20, 100, 500]
    n_epochs = len(epochs)
    #range(n_epochs)
    for j in epochs:
        for k in range(n_batch):
            b_indexes = batch_indexes_list[i][j][k]
            oob_indexes = X_indexes[X_indexes!=b_indexes]

            # batch
            plt.scatter(X[oob_indexes,0], y[oob_indexes], color="blue", alpha=0.05, label="data")
            plt.scatter(X[b_indexes,0], y[b_indexes], color="red", alpha=1.0, label="batch")
            plt.plot(x_range, previous_model, color="red")
            plt.title(f"Model predictions - batch_size={batch_size_list[i]}, epoch={j+1}, batch={k+1}")
            plt.xlabel("X")
            plt.ylabel("y")
            plt.savefig(f'{folderpath}plot_{cpt}.png')
            plt.close()
            
            cpt+=1

            # new weights
            plt.scatter(X[oob_indexes,0], y[oob_indexes], color="blue", alpha=0.05, label="data")
            plt.scatter(X[b_indexes,0], y[b_indexes], color="red", alpha=1.0, label="batch")
            plt.plot(x_range, previous_model, color="red", alpha=0.25) 
            w0, w1 = np.array(batch_weights_list[i])[j,k,0,0], np.array(batch_weights_list[i])[j,k,1,0]
            plt.plot(x_range, w1*x_norm_range + w0, color="red", alpha=1.0)
            previous_model = w1*x_norm_range + w0
            plt.title(f"Model predictions - batch_size={batch_size_list[i]}, epoch={j+1}, batch={k+1}")
            plt.xlabel("X")
            plt.ylabel("y")
            plt.savefig(f'{folderpath}plot_{cpt}.png')
            plt.close()

            cpt += 1
            
            print(f"\repoch={np.where(np.array(epochs) == j)[0][0]+1}/{n_epochs}, batch={k+1}/{n_batch} / images={cpt}/{1+2*n_epochs*n_batch}     ", end="")
    print()
            
    images = []
    for i in range(cpt):
        filename = f'{folderpath}plot_{i}.png'
        images.append(imageio.imread(filename))

    print("gif creation:", end="")
    imageio.mimsave(f'{folderpath}animated_plot.gif', images, duration=0.1, loop=0)
    print(" ok")


