import torch
import torch.nn as nn
import torch.nn.functional as F

class Grouper(nn.Module):
    def __init__(self, model, padder):
        super(Grouper, self).__init__()
        self.model = model
        self.padder = padder

    def forward(self, z, u):
        raise NotImplementedError

    def normalize_weights(self):
        with torch.no_grad():
            torch.clamp_(self.model.weight, min=0.0)
            w = self.model.weight
            assert (w.shape[-2] > 1 and w.shape[-3] > 1)
            norm_shape = (-1, w.shape[-1])
            norms = w.view(norm_shape).abs().sum([-1], keepdim=True)
            self.model.weight.view(norm_shape).div_(norms)



class Chi_Squared_from_Gaussian_1d(Grouper):
    def __init__(self, model, padder, trainable=False, mu_init=1.0, eps=1e-6):
        super(Chi_Squared_from_Gaussian_1d, self).__init__(model, padder)
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps

        nn.init.ones_(self.model.weight)
        self.nu = torch.numel(self.model.weight)

        if not trainable:
            self.model.weight.requires_grad = False

    def forward(self, z, u):      
        assert len(u.shape) == 4
        u_spatial = u.unsqueeze(1) ** 2.0

        u_spatial_padded = self.padder(u_spatial)
        v = self.model(u_spatial_padded).squeeze(1)

        std = 1.0 / torch.sqrt(v + self.eps)

        s = (z + self.correlated_mean_beta) * std.view(z.shape)
        return s


class Chi_Squared_from_Gaussian_2d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, trainable=False, mu_init=1, eps=1e-6):
        super(Chi_Squared_from_Gaussian_2d, self).__init__(model, padder)
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.spatial = int(cap_dim ** 0.5)

        nn.init.ones_(self.model.weight)

        if not trainable:
            self.model.weight.requires_grad = False

    def get_v(self, u):
        u_spatial = u.view(u.shape[0], 1, self.spatial, self.spatial, -1) ** 2.0
        u_spatial_padded = self.padder(u_spatial)
        v = self.model(u_spatial_padded).squeeze(1)
        return v

    def forward(self, z, u):      
        u_spatial = u.view(u.shape[0], 1, self.spatial, self.spatial, -1) ** 2.0
        u_spatial_padded = self.padder(u_spatial)
        v = self.model(u_spatial_padded).squeeze(1)

        std = 1.0 / torch.sqrt(v + self.eps)

        s =  (z + self.correlated_mean_beta) * std.view(z.shape)
        # s =  std.view(z.shape)

        return s


class Gaussian_2d_Test(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, trainable=False, mu_init=1, eps=1e-4):
        super(Gaussian_2d_Test, self).__init__(model, padder)
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.spatial = int(cap_dim ** 0.5)

        self.thresh = nn.Threshold(10, 10)

        nn.init.ones_(self.model.weight)

        if not trainable:
            self.model.weight.requires_grad = False

    def get_v(self, u):
        # u_spatial = F.relu(u.view(u.shape[0], 1, self.spatial, self.spatial, -1) + self.correlated_mean_beta) ** 2.0
        u_spatial = u.view(u.shape[0], 1, self.spatial, self.spatial, -1) ** 2.0
        u_spatial_padded = self.padder(u_spatial)
        v = self.model(u_spatial_padded).squeeze(1)
        return v

    def forward(self, z, u):      
        v = self.get_v(u)

        std = 1.0 / torch.sqrt(v + self.eps)
        # std = 1.0 / torch.sqrt(v + self.eps - self.correlated_mean_beta)

        # s =  (z + self.correlated_mean_beta) * std.view(z.shape)
        # s = z**2.0 * std.view(z.shape)
        s =  z + self.correlated_mean_beta * std.view(z.shape)
        # s = (self.correlated_mean_beta) * std.view(z.shape)
        # s = std.view(z.shape)# + self.correlated_mean_beta
        return s


class Chi_Squared_from_Gaussian_3d(Grouper):
    def __init__(self, model, padder, trainable=False, mu_init=1.0, eps=1e-6):
        super(Chi_Squared_from_Gaussian_3d, self).__init__(model, padder)
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps

        nn.init.ones_(self.model.weight)

        if not trainable:
            self.model.weight.requires_grad = False

    def forward(self, z, u):      
        spatial = int(round(u.shape[1] ** (1.0/3.0)))
        u_spatial = u.view(u.shape[0], 1, spatial, spatial, spatial) ** 2.0
        u_spatial_padded = self.padder(u_spatial)
        v = self.model(u_spatial_padded).squeeze(1)

        std = 1.0 / torch.sqrt(v + self.eps)

        s = (z + self.correlated_mean_beta) * std.view(z.shape)
        return s

    
class Stationary_Capsules_1d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, n_transforms,
                 mu_init=1.0, trainable=False, eps=1e-6):
        super(Stationary_Capsules_1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps

        nn.init.ones_(self.model.weight)
        if not trainable:
            self.model.weight.requires_grad = False

    def forward(self, z, u):
        h,w = u.shape[2], u.shape[3]
        u_caps = u.view(-1, self.n_t, self.n_caps, self.cap_dim, h*w) # (bsz, t, n_caps, capdim, h*w)
        u_caps = u_caps.permute((0, 2, 1, 3, 4)) # (bsz, n_caps, t, capdim, h*w)
        u_caps = u_caps.reshape(-1, 1, self.n_t, self.cap_dim, h*w)
        u_caps = u_caps ** 2.0
        u_caps_padded = self.padder(u_caps)
        v = self.model(u_caps_padded).squeeze(1)
        v = v.view(-1, self.n_caps, self.n_t, self.cap_dim, h*w)
        v = v.permute((0, 2, 1, 3, 4)) # (bsz, t, n_caps, capdim, h*w)

        v = v.reshape(z.shape)
        std = 1.0 / torch.sqrt(v + self.eps)
        s = (z + self.correlated_mean_beta) * std
        return s

    
class Chi_Squared_Capsules_from_Gaussian_1d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, n_transforms,
                 mu_init=1.0, n_off_diag=1, trainable=False, eps=1e-6):
        super(Chi_Squared_Capsules_from_Gaussian_1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps

        nn.init.zeros_(self.model.weight)
        with torch.no_grad():
            capdim = self.model.weight.shape[3]
            W = torch.ones([capdim, capdim])
            W = torch.triu(W, -n_off_diag)
            W = W*W.t()

            # For half/speed partial roll
            # W = W.repeat_interleave(repeats=2, dim=0).to('cuda')
            # for r in range(0, capdim):
            #     if r-2 >= 0:
            #         W[r*2, r-2] += 0.5
            #     if r+1 < capdim:
            #         W[r*2, r+1] -= 0.5

            self.model.weight.data[0, 0, :, :, 0] = W

        if not trainable:
            self.model.weight.requires_grad = False

    def get_v(self, u):
        h,w = u.shape[2], u.shape[3]
        u_caps = u.view(-1, self.n_t, self.n_caps, self.cap_dim, h*w) # (bsz, t, n_caps, capdim, h*w)
        u_caps = u_caps.permute((0, 2, 1, 3, 4)) # (bsz, n_caps, t, capdim, h*w)
        u_caps = u_caps.reshape(-1, 1, self.n_t, self.cap_dim, h*w)
        u_caps = u_caps ** 2.0
        u_caps_padded = self.padder(u_caps)
        v = self.model(u_caps_padded).squeeze(1)
        v = v.view(-1, self.n_caps, self.n_t, self.cap_dim, h*w)
        v = v.permute((0, 2, 1, 3, 4)) # (bsz, t, n_caps, capdim, h*w)
        return v

    def forward(self, z, u):
        v = self.get_v(u).reshape(z.shape)
        std = 1.0 / torch.sqrt(v + self.eps)
        s = (z + self.correlated_mean_beta) * std
        return s


class Partial_Seq_Capsules_from_Gaussian_1d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, n_transforms, bsz, n_t_out,
                 mu_init=1.0, n_off_diag=1, trainable=False, eps=1e-6):
        super(Partial_Seq_Capsules_from_Gaussian_1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps

        self.bsz = bsz
        self.n_t_out = n_t_out

        nn.init.zeros_(self.model.weight)
        with torch.no_grad():
            capdim = self.model.weight.shape[3]
            W = torch.ones([capdim, capdim])
            W = torch.triu(W, -n_off_diag)
            W = W*W.t()
            self.model.weight.data[0, 0, :, :, 0] = W

        if not trainable:
            self.model.weight.requires_grad = False

    def get_v(self, u):
        h,w = u.shape[2], u.shape[3]
        u_caps = u.view(-1, self.n_t, self.n_caps, self.cap_dim, h*w) # (bsz, t, n_caps, capdim, h*w)
        u_caps = u_caps.permute((0, 2, 1, 3, 4)) # (bsz, n_caps, t, capdim, h*w)
        u_caps = u_caps.reshape(-1, 1, self.n_t, self.cap_dim, h*w)
        u_caps = u_caps ** 2.0
        u_caps_padded = self.padder(u_caps)
        v = self.model(u_caps_padded).squeeze(1)
        v = v.view(-1, self.n_caps, self.n_t, self.cap_dim, h*w)
        v = v.permute((0, 2, 1, 3, 4)) # (bsz, t, n_caps, capdim, h*w)
        return v

    def forward(self, z, u):
        v = self.get_v(u).reshape(z.shape)
        std = 1.0 / torch.sqrt(v + self.eps)
        s = (z + self.correlated_mean_beta) * std
        s_crop = s.view(-1, self.n_t, self.n_caps, self.cap_dim, 1)[:, -self.n_t_out:, :, :, :].reshape(self.bsz*self.n_t_out, -1, 1, 1)
        return s_crop

class PartialSeq_NonTopographic_Capsules1d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, n_transforms, bsz, n_t_out,
                 mu_init=0, n_off_diag=0, trainable=False, eps=1e-6):
        super(PartialSeq_NonTopographic_Capsules1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.eps = eps
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.bsz = bsz
        self.n_t_out = n_t_out
       
    def forward(self, z, u):
        s = z
        s_crop = s.view(-1, self.n_t, self.n_caps, self.cap_dim, 1)[:, -self.n_t_out:, :, :, :].reshape(self.bsz*self.n_t_out, -1, 1, 1)
        return s_crop


class Causal_Capsules_from_Gaussian_1d(Grouper):
    def __init__(self, model, padder, n_caps, u_cap_dim, z_cap_dim, n_transforms,
                 mu_init=1.0, n_off_diag=1, trainable=False, eps=1e-6):
        super(Causal_Capsules_from_Gaussian_1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.u_cap_dim = u_cap_dim
        self.cap_dim = self.z_cap_dim = z_cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps
        
        wdim = self.model.weight.shape[3]
        self.n_u = wdim // 2
        self.z_cap_dim_cropped = wdim - self.n_u
        self.n_t_out = self.z_cap_dim_cropped

        nn.init.zeros_(self.model.weight)
        with torch.no_grad():
            W = torch.ones([wdim, wdim])
            W = torch.triu(W, -n_off_diag)
            W = W*W.t()

            W[self.n_u:, :] = 0.0
            W = W.tril()
            
            # For half/speed partial roll
            # W = W.repeat_interleave(repeats=2, dim=0).to('cuda')
            # for r in range(0, wdim):
            #     if r-2 >= 0:
            #         W[r*2, r-2] += 0.5
            #     if r+1 < capdim:
            #         W[r*2, r+1] -= 0.5

            self.model.weight.data[0, 0, :, :, 0] = W

        if not trainable:
            self.model.weight.requires_grad = False

    def forward(self, z, u):
        h,w = u.shape[2], u.shape[3]
        u_caps = u.view(-1, self.n_t, self.n_caps, self.u_cap_dim, h*w) # (bsz, t, n_caps, capdim, h*w)
        u_caps = u_caps.permute((0, 2, 1, 3, 4)) # (bsz, n_caps, t, capdim, h*w)
        u_caps = u_caps.reshape(-1, 1, self.n_t, self.u_cap_dim, h*w)
        u_caps = u_caps ** 2.0
        u_caps_padded = self.padder(u_caps)
        v = self.model(u_caps_padded).squeeze(1)
        v = v.view(-1, self.n_caps, self.n_t_out, self.z_cap_dim, h*w)
        v = v.permute((0, 2, 1, 3, 4)) # (bsz, t, n_caps, capdim, h*w)

        # z_rolled = torch.roll(z.view(-1, self.n_t, self.n_caps, self.z_cap_dim, h*w), shifts=self.n_u-1, dims=1).view(z.shape)
        # z_crop = z.view(-1, self.n_t, self.n_caps, self.z_cap_dim, h*w)[:, -(self.n_t_out+1):, :, :, :].reshape(v.shape[0]*self.n_t_out, -1, 1, 1)

        v = v.reshape(z.shape)
        std = 1.0 / torch.sqrt(v + self.eps)
        s = (z + self.correlated_mean_beta) * std
        return s


class LocalSum_Capsules_from_Gaussian_1d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, n_transforms,
                 mu_init=1.0, n_off_diag=1, trainable=False, eps=1e-6):
        super(LocalSum_Capsules_from_Gaussian_1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps

        nn.init.zeros_(self.model.weight)
        with torch.no_grad():
            capdim = self.model.weight.shape[3]
            W = torch.ones([capdim, capdim])
            W = torch.triu(W, -n_off_diag)
            W = W*W.t()

            # For half/speed partial roll
            # W = W.repeat_interleave(repeats=2, dim=0).to('cuda')
            # for r in range(0, capdim):
            #     if r-2 >= 0:
            #         W[r*2, r-2] += 0.5
            #     if r+1 < capdim:
            #         W[r*2, r+1] -= 0.5

            self.model.weight.data[0, 0, :, :, 0] = W

        if not trainable:
            self.model.weight.requires_grad = False

    def forward(self, z, u):
        h,w = u.shape[2], u.shape[3]
        u_caps = u.view(-1, self.n_t, self.n_caps, self.cap_dim, h*w) # (bsz, t, n_caps, capdim, h*w)
        u_caps = u_caps.permute((0, 2, 1, 3, 4)) # (bsz, n_caps, t, capdim, h*w)
        u_caps = u_caps.reshape(-1, 1, self.n_t, self.cap_dim, h*w)

        # u_caps = u_caps ** 2.0

        u_caps_padded = self.padder(u_caps)
        v = self.model(u_caps_padded).squeeze(1)
        v = v.view(-1, self.n_caps, self.n_t, self.cap_dim, h*w)
        v = v.permute((0, 2, 1, 3, 4)) # (bsz, t, n_caps, capdim, h*w)

        v = v.reshape(z.shape)
        # std = 1.0 / torch.sqrt(v + self.eps)
        # s = (z + self.correlated_mean_beta) * std
        s = v
        return s


class NonTopographic_Capsules1d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, n_transforms,
                 mu_init=0, n_off_diag=0, trainable=False, eps=1e-6):
        super(NonTopographic_Capsules1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.eps = eps
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
       
    def forward(self, z, u):
        s = z
        return s



class Chi_Squared_Capsules_from_Gaussian_2d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim1, cap_dim2, n_transforms,
                  mu_init=1.0, n_off_diag=1, trainable=False, eps=1e-6):
        super(Chi_Squared_Capsules_from_Gaussian_2d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim1 = cap_dim1
        self.cap_dim2 = cap_dim2
        self.n_t = n_transforms
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps
        
        nn.init.ones_(self.model.weight)
        with torch.no_grad():
            d = self.model.weight.shape[3]
            W = torch.ones([d, d])
            W = torch.triu(W, -n_off_diag)
            W = W*W.t()

        if not trainable:
            self.model.weight.requires_grad = False

        self.nu = (self.model.weight > 0.0).sum()


    def forward(self, z, u):
        h,w = u.shape[2], u.shape[3]
        u_caps = u.view(-1, self.n_t, self.n_caps, self.cap_dim1, self.cap_dim2) # (bsz, t, n_caps, c1, c2)
        u_caps = u_caps.permute((0, 2, 1, 3, 4)) # (bsz, n_caps, t, c1, c2)
        u_caps = u_caps.reshape(-1, 1, self.n_t, self.cap_dim1, self.cap_dim2)
        u_caps = u_caps ** 2.0

        u_caps_padded = self.padder(u_caps)
        v1 = self.model(u_caps_padded)

        u_caps_d2 = u_caps.permute((0, 1, 2, 4, 3)) # (bsz, n_caps, t, c2, c1)
        u_caps_d2_padded = self.padder(u_caps_d2)
        v2 = self.model(u_caps_d2_padded)
        v2 = v2.permute((0, 1, 2, 4, 3))  # (bsz, n_caps, t, c1, c2)

        v1 = v1.view(-1, self.n_caps, self.n_t, self.cap_dim1, self.cap_dim2)
        v1 = v1.permute((0, 2, 1, 3, 4)) # (bsz, t, n_caps, c1, c2)
        v1 = v1.reshape(z.shape)

        v2 = v2.view(-1, self.n_caps, self.n_t, self.cap_dim1, self.cap_dim2)
        v2 = v2.permute((0, 2, 1, 3, 4)) # (bsz, t, n_caps, c1, c2)
        v2 = v2.reshape(z.shape)

        s = (z + self.correlated_mean_beta) / (torch.sqrt(v1 + self.eps) + torch.sqrt(v2 + self.eps))

        return s




class Chi_Squared_Capsules_from_Gaussian_3d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim1, cap_dim2, cap_dim3, n_transforms,
                 mu_init=1.0, n_off_diag=1, trainable=False, eps=1e-6):
        super(Chi_Squared_Capsules_from_Gaussian_3d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim1 = cap_dim1
        self.cap_dim2 = cap_dim2
        self.cap_dim3 = cap_dim3
        self.n_t = n_transforms
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps

        nn.init.ones_(self.model.weight)
        # Banded init
        with torch.no_grad():
            d = self.model.weight.shape[3]
            W = torch.ones([d, d])
            W = torch.triu(W, -n_off_diag)
            W = W*W.t()

        if not trainable:
            self.model.weight.requires_grad = False

    def forward(self, z, u):
        h,w = u.shape[2], u.shape[3]
        u_caps = u.view(-1, self.n_t, self.n_caps, self.cap_dim1, self.cap_dim2, self.cap_dim3) # (bsz, t, n_caps, c1, c2, c3)
        u_caps = u_caps.permute((0, 2, 1, 3, 4, 5)) # (bsz, n_caps, t, c1, c2, c3)
        u_caps = u_caps.reshape(-1, 1, self.n_t, self.cap_dim1, self.cap_dim2, self.cap_dim3) # (bsz*n_caps, 1, t, c1, c2, c3)
        u_caps = u_caps ** 2.0

        u_caps_d1 = u_caps.permute((0, 1, 5, 2, 3, 4)) #  (bsz*n_caps, 1, t, c1, c2, c3) --> (bsz*n_caps, 1, c3, t, c1, c2)
        u_caps_d1 = u_caps_d1.reshape(-1, 1, self.n_t, self.cap_dim1, self.cap_dim2) # (bsz*n_caps*c3, 1, t, c1, c2)
        u_caps_d1_padded = self.padder(u_caps_d1)
        v1 = self.model(u_caps_d1_padded)
        v1 = v1.reshape(-1, self.n_caps, self.cap_dim3, self.n_t, self.cap_dim1, self.cap_dim2) 
        v1 = v1.permute((0, 3, 1, 4, 5, 2)).reshape(z.shape) # (bsz, n_caps, c3, t, c1, c2)--> (bsz, t, n_caps, c1, c2, c3)

        u_caps_d2 = u_caps.permute((0, 1, 3, 2, 4, 5)) #  (bsz*n_caps, 1, t, c1, c2, c3) --> (bsz*n_caps, 1, c1, t, c2, c3)
        u_caps_d2 = u_caps_d2.reshape(-1, 1, self.n_t, self.cap_dim2, self.cap_dim3) # (bsz*n_caps*c1, 1, t, c2, c3)
        u_caps_d2_padded = self.padder(u_caps_d2)
        v2 = self.model(u_caps_d2_padded)
        v2 = v2.reshape(-1, self.n_caps, self.cap_dim1, self.n_t, self.cap_dim2, self.cap_dim3) # (bsz, n_caps, c1, t, c2, c3)
        v2 = v2.permute((0, 3, 1, 2, 4, 5)).reshape(z.shape) # (bsz, n_caps, c1, t, c2, c3) --> (bsz, t, n_caps, c1, c2, c3)

        u_caps_d3 = u_caps.permute((0, 1, 3, 2, 5, 4)) # (bsz*n_caps, 1, t, c1, c2, c3) --> (bsz*n_caps, 1, c1, t, c3, c2)
        u_caps_d3 = u_caps_d3.reshape(-1, 1, self.n_t, self.cap_dim3, self.cap_dim2) # (bsz*n_caps*c1, 1, t, c3, c2)
        u_caps_d3_padded = self.padder(u_caps_d3)
        v3 = self.model(u_caps_d3_padded)
        v3 = v3.reshape(-1, self.n_caps, self.cap_dim1, self.n_t, self.cap_dim3, self.cap_dim2) # (bsz, n_caps, c1, t, c3, c2)
        v3 = v3.permute((0, 3, 1, 2, 5, 4)).reshape(z.shape) # (bsz, n_caps, c1, t, c3, c2) --> (bsz, t, n_caps, c1, c2, c3)


        s = (z + self.correlated_mean_beta) / (torch.sqrt(v1 + self.eps) 
                                                   + torch.sqrt(v2 + self.eps) 
                                                   + torch.sqrt(v3 + self.eps))

        return s
