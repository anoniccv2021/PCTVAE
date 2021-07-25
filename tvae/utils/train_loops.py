import os
import torch
from tqdm import tqdm
from tvae.utils.vis import (Plot_TransformSelectivity, plot_recon, 
                            Plot_MaxActImg, Plot_ClassActMap, Plot_AllCapOffsets,
                            plot_activation_stats)
from tvae.utils.correlations import Plot_Covariance_Matrix
from tvae.utils.losses import all_pairs_equivariance_loss, get_cap_offsets, get_allcap_offsets
import numpy as np

def train_epoch(model, optimizer, train_loader, log, savepath, epoch, eval_batches=300,
                plot_weights=False, plot_fullcaptrav=False, wandb_on=True):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    total_eq_loss = 0.0
    num_batches = 0

    model.train()
    for x, label in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.float().to('cuda')
        # x = torch.tanh(x)

        x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
        z, u, s, probs_x, kl_z, kl_u, neg_logpx_z = model(x_batched)

        avg_KLD = (kl_z.sum() + kl_u.sum()) / kl_u.shape[0]
        avg_neg_logpx_z = neg_logpx_z.sum() / neg_logpx_z.shape[0]
        loss = avg_neg_logpx_z + avg_KLD
        
        eq_loss = all_pairs_equivariance_loss(s.detach(), bsz=x.shape[0], 
                                              seq_len=x.shape[1], n_caps=model.grouper.n_caps,
                                              cap_dim=model.grouper.cap_dim)

        loss.backward()
        optimizer.step()

        # model.normalize_weights(including_grouper=False)

        total_loss += loss
        total_neg_logpx_z += avg_neg_logpx_z
        total_kl += avg_KLD
        total_eq_loss += eq_loss
        num_batches += 1
        b_idx = epoch * len(train_loader) + num_batches

        if b_idx % eval_batches == 0:
            log('Train Total Loss', loss)
            log('Train -LogP(x|z)', avg_neg_logpx_z)
            log('Train KLD', avg_KLD)
            log('Eq Loss', eq_loss)

            if plot_weights:
                model.plot_decoder_weights(wandb_on=wandb_on)
                model.plot_encoder_weights(wandb_on=wandb_on)

            Plot_Covariance_Matrix(s**2.0, s**2.0, name='Covariance_S**2_batch', wandb_on=wandb_on)

            if plot_fullcaptrav:
                model.plot_capsule_traversal(x_batched.detach(), 
                                             os.path.join(savepath, 'samples'),
                                             b_idx, wandb_on=wandb_on)
            
            plot_recon(x_batched, 
                       probs_x.view(x_batched.shape), 
                       os.path.join(savepath, 'samples'),
                       b_idx, wandb_on=wandb_on)

    return total_loss, total_neg_logpx_z, total_kl, total_eq_loss, num_batches


def eval_epoch(model, val_loader, log, savepath, epoch, n_is_samples=100, 
               plot_maxact=False, plot_class_selectivity=False, 
               plot_cov=False, wandb_on=True, plot_fullcaptrav=False):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    total_is_estimate = 0.0
    total_eq_loss = 0.0
    num_batches = 0
    all_x = []
    all_s = []
    all_u = []
    all_z = []
    all_v = []

    all_labels = []

    model.eval()
    with torch.no_grad():
        for x, label in tqdm(val_loader):
            x = x.float().to('cuda')

            x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
            z, u, s, probs_x, kl_z, kl_u, neg_logpx_z = model(x_batched)
            avg_KLD = (kl_z.sum() + kl_u.sum()) / kl_u.shape[0]
            avg_neg_logpx_z = neg_logpx_z.sum() / neg_logpx_z.shape[0]

            eq_loss = all_pairs_equivariance_loss(s, bsz=x.shape[0], seq_len=x.shape[1], 
                                                  n_caps=model.grouper.n_caps, cap_dim=model.grouper.cap_dim)

            # all_s.append(s.cpu().detach())
            # all_z.append(z.cpu().detach())
            # all_u.append(u.cpu().detach())
            # all_v.append(model.grouper.get_v(u).cpu().detach())

            all_x.append(x.cpu().detach())
            all_labels.append(label.cpu().detach())

            loss = avg_neg_logpx_z + avg_KLD
            total_loss += loss
            total_neg_logpx_z += avg_neg_logpx_z
            total_kl += avg_KLD
            total_eq_loss += eq_loss

            is_estimate = model.get_IS_estimate(x_batched, n_samples=n_is_samples)
            total_is_estimate += is_estimate.sum() / neg_logpx_z.shape[0]

            num_batches += 1

    if plot_cov or plot_maxact or plot_class_selectivity:
        all_s = torch.cat(all_s, 0)
        all_z = torch.cat(all_z, 0)
        all_u = torch.cat(all_u, 0)
        all_v = torch.cat(all_v, 0)

        all_x = torch.cat(all_x, 0)
        all_labels = torch.cat(all_labels, 0)
    if plot_cov:
        Plot_Covariance_Matrix(all_s, all_s, name='Covariance_S_Full', wandb_on=wandb_on)
        Plot_Covariance_Matrix(all_s**2.0, all_s**2.0, name='Covariance_S**2_Full', wandb_on=wandb_on)
    if plot_maxact:
        Plot_MaxActImg(all_s, all_x, os.path.join(savepath, 'samples'), epoch, wandb_on=wandb_on)
    if plot_class_selectivity:
        Plot_ClassActMap(all_s, all_labels, os.path.join(savepath, 'samples'), epoch, wandb_on=wandb_on)

    if plot_fullcaptrav:
        model.plot_capsule_traversal(x_batched.detach(), 
                                        os.path.join(savepath, 'samples'),
                                        0, wandb_on=wandb_on)
    # plot_activation_stats(all_z, all_u, all_s, all_v, model.grouper.correlated_mean_beta)

    return total_loss, total_neg_logpx_z, total_kl, total_is_estimate, total_eq_loss, num_batches




def final_results(model, val_loader, log, n_is_samples=10, dt_set=range(0, 18),
                    wandb_on=True):
    total_is_estimate_by_dt = {i: 0.0 for i in dt_set}
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for x, label in tqdm(val_loader):
            x = x.float().to('cuda')
            x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
            for dt in dt_set:
                is_estimate = model.get_IS_estimate(x_batched, n_samples=n_is_samples, delta_t=dt)
                total_is_estimate_by_dt[dt] += is_estimate.sum() / x_batched.shape[0]
            num_batches += 1

    for dt in total_is_estimate_by_dt:
        total_is_estimate_by_dt[dt] = total_is_estimate_by_dt[dt] / num_batches

    return total_is_estimate_by_dt





def train_epoch_dsprites(model, optimizer, train_loader, log, savepath, epoch, eval_batches=300,
                         plot_weights=False, plot_fullcaptrav=False, 
                         plot_transform_selectivity=False, plot_allcapoffsets=False,
                         compute_disentanglement=False, wandb_on=True):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    total_eq_loss = 0.0
    allcap_offsets = {idx:[[] for _ in range(model.grouper.n_caps)] for idx in range(5)}
    alltrue_offsets = {idx:[[] for _ in range(model.grouper.n_caps)] for idx in range(5)}
    cap_offsets = {idx:[] for idx in range(5)}
    true_offsets = {idx:[] for idx in range(5)}
    all_s = []
    all_transforms = []
    num_batches = 0

    model.train()
    for x, label in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.float().to('cuda')

        x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
        z, u, s, probs_x, kl_z, kl_u, neg_logpx_z = model(x_batched)

        # v = model.grouper.get_v(z,u)
        # reweights = v.flatten(start_dim=1).sum(-1) / (v.sum() / v.shape[0])

        # KLD = kl_z.flatten(start_dim=1).sum(-1) + kl_u.flatten(start_dim=1).sum(-1)
        # neg_logpx_z = neg_logpx_z.flatten(start_dim=1).sum(-1)
        # loss = ((KLD + neg_logpx_z) * reweights).mean()

        avg_KLD = (kl_z.sum() + kl_u.sum()) / x_batched.shape[0]
        avg_neg_logpx_z = neg_logpx_z.sum() / x_batched.shape[0]
        loss = avg_neg_logpx_z + avg_KLD

        eq_loss = all_pairs_equivariance_loss(s.detach(), bsz=x.shape[0], seq_len=x.shape[1],
                                              n_caps=model.grouper.n_caps, cap_dim=model.grouper.cap_dim)
   
        if plot_transform_selectivity:
            t_idxs = torch.argmax(torch.abs(label[:, 0] - label[:, 1]), dim=1)
            all_transforms.append(t_idxs.cpu().detach())
            # Only need first element of sequence of s's 
            all_s.append(s.view(x.shape[0], -1, *s.shape[1:])[:, 0].cpu().detach())
    
        if compute_disentanglement:
            cap_offsets, true_offsets = get_cap_offsets(cap_offsets, true_offsets, s, 
                                                        label, bsz=x.shape[0], seq_len=x.shape[1], 
                                                        n_caps=model.grouper.n_caps, cap_dim=model.grouper.cap_dim)
        
        if plot_allcapoffsets:
            allcap_offsets, alltrue_offsets = get_allcap_offsets(allcap_offsets, alltrue_offsets, s,
                                                            label, bsz=x.shape[0], seq_len=x.shape[1], 
                                                            n_caps=model.grouper.n_caps, cap_dim=model.grouper.cap_dim)

        loss.backward()
        optimizer.step()    

        # model.normalize_weights(including_grouper=False)

        total_loss += loss
        total_neg_logpx_z += avg_neg_logpx_z
        total_kl += avg_KLD
        total_eq_loss += eq_loss
        num_batches += 1
        b_idx = epoch * len(train_loader) + num_batches

        if b_idx % eval_batches == 0:
            log('Train Total Loss', loss)
            log('Train -LogP(x|z)', avg_neg_logpx_z)
            log('Train KLD', avg_KLD)
            log('Eq Loss', eq_loss)

            if plot_weights:
                model.plot_decoder_weights(wandb_on=wandb_on)
                model.plot_encoder_weights(wandb_on=wandb_on)

            Plot_Covariance_Matrix(s**2.0, s**2.0, name='Covariance_S**2_batch', wandb_on=wandb_on)

            if plot_fullcaptrav:
                model.plot_capsule_traversal(x_batched.detach(), 
                                             os.path.join(savepath, 'samples'),
                                             b_idx, wandb_on=wandb_on)

            if plot_transform_selectivity:
                Plot_TransformSelectivity(all_s, all_transforms, os.path.join(savepath, 'samples'), epoch, 
                                          name='Train Transform Selectivity', n_caps=model.grouper.n_caps,
                                          cap_dim=model.grouper.cap_dim, wandb_on=wandb_on)

            plot_recon(x_batched, 
                       probs_x.view(x_batched.shape), 
                       os.path.join(savepath, 'samples'),
                       b_idx, wandb_on=wandb_on)

            if compute_disentanglement:
                for t in cap_offsets:
                    log(f'Train Dist Corr {t}', np.corrcoef(cap_offsets[t], true_offsets[t])[0,1])

            if plot_allcapoffsets:
                Plot_AllCapOffsets(allcap_offsets, alltrue_offsets, os.path.join(savepath, 'samples'), epoch, 
                                   name='Train AllCapOffsets', wandb_on=wandb_on)
    
    total_dist_corr = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    if compute_disentanglement:
        for t in cap_offsets:
            total_dist_corr[t] = np.corrcoef(cap_offsets[t], true_offsets[t])[0,1]

    if plot_allcapoffsets:
        Plot_AllCapOffsets(allcap_offsets, alltrue_offsets, os.path.join(savepath, 'samples'), epoch, 
                                   name='Epoch AllCapOffsets', wandb_on=wandb_on)

    if plot_transform_selectivity:
        Plot_TransformSelectivity(all_s, all_transforms, os.path.join(savepath, 'samples'), epoch, 
                                  name='Epoch Transform Selectivity', n_caps=model.grouper.n_caps,
                                  cap_dim=model.grouper.cap_dim, wandb_on=wandb_on)

    return total_loss, total_neg_logpx_z, total_kl, total_eq_loss, total_dist_corr, num_batches

