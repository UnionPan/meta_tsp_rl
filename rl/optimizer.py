import torch


def conjugate_gradient(Avp, b, nsteps, residual_tol=1e-10):
    "do conjugate gradient to find an approximated v such that A v = b"

    x = torch.zeros(b.size())
    r = b - Avp(x)
    p = r
    rdotr = torch.dot(r, r)

    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def line_search(policy_net, get_loss, full_step, grad, max_num_backtrack=10, accept_ratio=0.1):
    """
    do backtracking line search
    ref: https://en.wikipedia.org/wiki/Backtracking_line_search
    :param policy_net: policy net used to get initial params and set params before get_loss
    :param get_loss: get loss evaluation
    :param full_step: maximum stepsize, numpy.ndarray
    :param grad: initial gradient i.e. nabla f(x) in wiki
    :param max_num_backtrack: maximum iterations of backtracking
    :param accept_ratio: i.e. param c in wiki
    :return: a tuple (whether accepted at last, found optimal x)
    """
    # initial point
    x0 = policy_net.get_flat_params()
    # initial loss
    f0 = get_loss(None)
    # step fraction
    alpha = 1.0
    # expected maximum improvement, i.e. cm in wiki
    expected_improve = accept_ratio * \
        (- torch.Tensor(full_step) * grad).sum(0, keepdim=True)

    for count in range(max_num_backtrack):
        xnew = x0 + alpha * full_step
        policy_net.set_flat_params(xnew)
        fnew = get_loss(None)
        actual_improve = f0 - fnew
        if actual_improve > 0 and actual_improve > alpha * expected_improve:
            return True, xnew
        alpha *= 0.5
    return False, x0
