import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import wandb
from neuralpredictors.measures import modules
from neuralpredictors.training import LongCycler, early_stopping
from nnfabrik.utility.nn_helpers import set_random_seed
from tqdm import tqdm
from torch.func import jacfwd, vmap

from ..utility import scores
from ..utility.scores import get_correlations, get_poisson_loss


## TODO - add loading from checkpoints in case of train stop
def standard_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    cb=None,
    detach_core=False,
    core_state_dict=None,
    use_wandb=True,
    wandb_project="factorised_core_parameter_search",
    wandb_entity="movies_parameter_search",
    wandb_name=None,
    wandb_model_config=None,
    wandb_dataset_config=None,
    print_step=1000,
    save_checkpoints=True,
    checkpoint_save_path="local/",
    chpt_save_step=15,
    deeplake_ds=False,
    validation_str = "oracle", # or "validation"
    unit_speed_flag=False,
    unit_speed_scale=0,
    **kwargs,
):
    """

    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        avg_loss: whether to average (or sum) the loss over a batch
        scale_loss: whether to scale the loss according to the size of the dataset
        loss_function: loss function to use
        stop_function: the function (metric) that is used to determine the end of the training in early stopping
        loss_accum_batch_n: number of batches to accumulate the loss over
        device: device to run the training on
        verbose: whether to print out a message for each optimizer step
        interval: interval at which objective is evaluated to consider early stopping
        patience: number of times the objective is allowed to not become better before the iterator terminates
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of training iterations
        maximize: whether to maximize or minimize the objective function
        tolerance: tolerance for early stopping
        restore_best: whether to restore the model to the best state after early stopping
        lr_decay_steps: how many times to decay the learning rate after no improvement
        lr_decay_factor: factor to decay the learning rate with
        min_lr: minimum learning rate
        cb: whether to execute callback function
        core_state_dict: if detach_core=True, we would use transfer learning, and the core is borrowed from this state_dict (.pth) file
        **kwargs:

    Returns:

    """

    def full_objective(model, dataloader, data_key, *args, **kwargs):
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        # todo - think how to avoid sum in model.core.regularizer()
        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(
                not detach_core
            ) * model.core.regularizer() + model.readout.regularizer(data_key)
        else:
            regularizers = int(not detach_core) * sum(
                model.core.regularizer()
            ) + model.readout.regularizer(data_key)
        if deeplake_ds:
            for k in kwargs.keys():
                if k not in ["id", "index"]:
                    kwargs[k] = torch.Tensor(np.asarray(kwargs[k])).to(device)
        model_output = model(args[0].to(device), data_key=data_key, **kwargs)
        time_left = model_output.shape[1]

        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)

        # unit speed loss
        unitspeedloss = torch.zeros(1).to(device)
        if kwargs['unit_speed_flag']:
            # model.readout[data_key].feature_latent.shape: (num_of_neuron, feature_latent_dim)
            # model.readout[data_key].features.shape: (1, channel_num, 1, num_of_neuron)
            # inputs: (m, d) tensor
            mm, dd = model.readout[data_key].feature_latent.shape
            # outputs tensor
            nnn = model.readout[data_key].features.shape[1]
            # Compute Jacobian for each data point
            # jacobians = []
            # for ii in range(mm):
            #     J_i = torch.autograd.functional.jacobian(lambda x: model.readout[data_key].features[0, :, 0, ii], model.readout[data_key].feature_latent[ii,:])
            #     jacobians.append(J_i)
            # jacobians = torch.stack(jacobians)  # Shape: (m, n, d)
            t_values = model.readout[data_key].feature_latent
            if model.readout[data_key].position_encoding_flag:
                def g(x):
                    result = model.readout[data_key].pos_enc_func(x)
                    return result, result
                J1, t_values = vmap(jacfwd(g, has_aux=True))(t_values)
                J1 = J1.squeeze()
                # print (f'J1.shape: {J1.shape}')
            jacobians = vmap(jacfwd(model.readout[data_key].feature_mlp))(t_values)
            jacobians = jacobians.squeeze()  # # Shape: (m, n, d)
            # print (f'jacobians.shape: {jacobians.shape}')
            if model.readout[data_key].position_encoding_flag:
                jacobians = jacobians @ J1
            # print (f'jacobians.shape: {jacobians.shape}')
            # assert False
            # Compute metric tensor G_i = J_i^T @ J_i for each data point
            G = torch.einsum('mnd,mne->mde', jacobians, jacobians)
            # Create identity matrix replicated across all samples (m, d, d)
            E = torch.eye(dd).unsqueeze(0).expand(mm, -1, -1).to(G.device)
            # Compute MSE loss between G and E
            unitspeedloss = nn.functional.mse_loss(G, E)

        return (
            loss_scale
            * criterion(
                model_output,
                original_data,
            )
            + regularizers
            + unitspeedloss * kwargs['unit_speed_scale']
        )

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders[validation_str],
        device=device,
        per_neuron=False,
        avg=True,
        deeplake_ds=deeplake_ds,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )
    print(f"optim_step_count = {optim_step_count}")

    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr_init,
                "architecture": wandb_model_config,
                "dataset": wandb_dataset_config,
                "cur_epochs": max_iter,
                "starting epoch": epoch,
                "lr_decay_steps": lr_decay_steps,
                "lr_decay_factor": lr_decay_factor,
                "min_lr": min_lr,
            },
        )

        wandb.define_metric(name="Epoch", hidden=True)
        wandb.define_metric(name="Batch", hidden=True)

    batch_no_tot = 0
    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):
        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0
        epoch_val_loss = 0
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            batch_args = list(data)

            batch_kwargs = data._asdict() if not isinstance(data, dict) else data

            loss = full_objective(
                model,
                dataloaders["train"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
                unit_speed_flag=unit_speed_flag,
                unit_speed_scale=unit_speed_scale,
            )
            loss = loss/optim_step_count
            loss.backward()

            epoch_loss += loss.detach()
            if (batch_no + 1) % optim_step_count == 0: # TODO: or (batch_no + 1 == len(LongCycler(dataloaders["train"])))
                optimizer.step()

                #                 optimizer.zero_grad(set_to_none=False)
                optimizer.zero_grad(set_to_none=True)
        
        model.eval()
        #yqiu
        #if save_checkpoints:
        #    if epoch % chpt_save_step == 0:
        #        torch.save(
        #            model.state_dict(), f"{checkpoint_save_path}epoch_{epoch}.pth"
        #        )

        ## after - epoch-analysis

        validation_correlation = get_correlations(
            model,
            dataloaders[validation_str],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=deeplake_ds,
        )
        val_loss = full_objective(
            model,
            dataloaders[validation_str],
            data_key,
            *batch_args,
            **batch_kwargs,
            detach_core=detach_core,
            unit_speed_flag=unit_speed_flag,
            unit_speed_scale=unit_speed_scale,
        )
        print(
            f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation}")

        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                #yqiu
                #"Batch": batch_no_tot, 
                #"Epoch": epoch,
                "validation_correlation": validation_correlation,
                "Epoch validation loss": val_loss,
                "Epoch": epoch,
            }
            wandb.log(wandb_dict)
        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    if save_checkpoints:
        #yqiu
        #torch.save(model.state_dict(), f"{checkpoint_save_path}final.pth")
        torch.save(model.state_dict(), f"{checkpoint_save_path}.pth")

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloaders[validation_str], device=device, as_dict=False, per_neuron=False, deeplake_ds=deeplake_ds,
    )
    print(f"\n\n FINAL validation_correlation {validation_correlation} \n\n")

    output = {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)
    if use_wandb:
        wandb.finish()

    # removing the checkpoints except the last one
    #yqiu, comment these lines
    #to_clean = os.listdir(checkpoint_save_path)
    #for f2c in to_clean:
    #    if "epoch_" in f2c and f2c[-4:] == ".pth":
    #        os.remove(f"{checkpoint_save_path}{f2c}")

    return score, output, model.state_dict()
