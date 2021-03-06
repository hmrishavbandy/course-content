def elbo(x, phi, density_net, sig_x, n):
    zs = rsample(phi, n)
    # Density net expects just [b,k] inputs, so we'll collapse together batch 
    # and samples dimensions to get [b*n,k] samples of z, then expand back out
    # separate [b,n,p] dimensions in the result
    b = x.size()[0]
    mu_xs = density_net(zs.view(b*n, -1)).view(b,n,-1)
    return log_p_x(x, mu_xs, sig_x) - kl_q_p(zs, phi)
