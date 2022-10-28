import deepxde as dde


def NS2D(Rey):
    """2D Navier Stokes PDE.

    Argument:
        - Reynolds number
    """
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        # p = V[:,2:3]

        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy),
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy),
        ]

    return pde


def RANS2D(Rey):
    """2D RANS PDE with Reynolds stresses.

    Argument:
        - Reynolds number
    """
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]

        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        # Reynolds stresses
        duu_x = dde.grad.jacobian(V, X, i=3, j=0)
        duv_x = dde.grad.jacobian(V, X, i=5, j=0)
        dvv_y = dde.grad.jacobian(V, X, i=4, j=1)
        duv_y = dde.grad.jacobian(V, X, i=5, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey *
            (du_xx + du_yy) - duu_x - duv_x,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey *
            (dv_xx + dv_yy) - dvv_y - duv_y,
        ]

    return pde


def RANSf2D(Rey):
    """2D RANS PDE with RANS forcing.
    Forcing is split into solenoidal component and potential component.

    Argument:
        - Reynolds number
    """
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        fsx = V[:, 3:4]
        fsy = V[:, 4:5]
        psi = V[:, 5:6]
        ppsi = p+psi

        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dppsi_x = dde.grad.jacobian(ppsi[:, None], X, i=0, j=0)
        dppsi_y = dde.grad.jacobian(ppsi[:, None], X, i=0, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dfsx_x = dde.grad.jacobian(V, X, i=3, j=0)
        dfsy_y = dde.grad.jacobian(V, X, i=4, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dppsi_x - 1.0 / Rey * (du_xx + du_yy) + fsx,
            u * dv_x + v * dv_y + dppsi_y - 1.0 / Rey * (dv_xx + dv_yy) + fsy,
            dfsx_x + dfsy_y
        ]
    return pde


def RANSf02D(Rey):
    """2D RANS PDE with RANS forcing.

    Forcing is solenoidal - the potential component is not modelled
    by the network and implicitly lumped together with pressure.

    Argument:
        - Reynolds number
    """
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        fsx = V[:, 3:4]
        fsy = V[:, 4:5]

        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dfsx_x = dde.grad.jacobian(V, X, i=3, j=0)
        dfsy_y = dde.grad.jacobian(V, X, i=4, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) + fsx,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) + fsy,
            dfsx_x + dfsy_y
        ]

    return pde


def RANSf0var2D(Rey):
    """2D RANS PDE with RANS forcing.
    Curl of the forcing is one of the outputs of the network.


    Argument:
        - Reynolds number
    """
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        fsx = V[:, 3:4]
        fsy = V[:, 4:5]
        curlf = V[:, 5:6]

        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dfsx_x = dde.grad.jacobian(V, X, i=3, j=0)
        dfsy_y = dde.grad.jacobian(V, X, i=4, j=1)
        dfsx_y = dde.grad.jacobian(V, X, i=3, j=1)
        dfsy_x = dde.grad.jacobian(V, X, i=4, j=0)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) + fsx,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) + fsy,
            dfsx_x + dfsy_y,
            curlf - (dfsy_x-dfsx_y)
        ]

    return pde


def RANSpknown2D(Rey):
    """2D RANS PDE with RANS forcing assuming p is given everywhere.

    Argument:
        - Reynolds number
    """
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        fsx = V[:, 3:4]
        fsy = V[:, 4:5]
        # p = V[:,2:3]

        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        dfsx_x = dde.grad.jacobian(V, X, i=3, j=0)
        dfsy_y = dde.grad.jacobian(V, X, i=4, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 / Rey * (du_xx + du_yy) + fsx,
            u * dv_x + v * dv_y + dp_y - 1.0 / Rey * (dv_xx + dv_yy) + fsy
        ]

    return pde


def RANSReStresses2D(Rey):
    """2D RANS PDE with Reynolds stresses.

    uu and vv are soft-constraint to be positive.

    Argument:
        - Reynolds number
    """
    def pde(X, V):
        u = V[:, 0:1]
        v = V[:, 1:2]
        p = V[:, 2:3]
        uu = V[:, 3:4]
        uv = V[:, 4:5]
        vv = V[:, 5:6]

        # p = V[:,2:3]
        du_x = dde.grad.jacobian(V, X, i=0, j=0)
        dv_y = dde.grad.jacobian(V, X, i=1, j=1)
        du_y = dde.grad.jacobian(V, X, i=0, j=1)
        dv_x = dde.grad.jacobian(V, X, i=1, j=0)
        dp_x = dde.grad.jacobian(V, X, i=2, j=0)
        dp_y = dde.grad.jacobian(V, X, i=2, j=1)
        du_xx = dde.grad.hessian(u, X, i=0, j=0)
        dv_xx = dde.grad.hessian(v, X, i=0, j=0)
        du_yy = dde.grad.hessian(u, X, i=1, j=1)
        dv_yy = dde.grad.hessian(v, X, i=1, j=1)

        duu_x = dde.grad.jacobian(V, X, i=3, j=0)
        duv_y = dde.grad.jacobian(V, X, i=4, j=1)
        duv_x = dde.grad.jacobian(V, X, i=4, j=0)
        dvv_y = dde.grad.jacobian(V, X, i=5, j=1)

        return [
            du_x + dv_y,
            u * du_x + v * du_y + dp_x - 1.0 /
            Rey * (du_xx + du_yy) + duu_x+duv_y,
            u * dv_x + v * dv_y + dp_y - 1.0 /
            Rey * (dv_xx + dv_yy) + duv_x+dvv_y,
            dde.backend.tf.nn.relu(-uu) + dde.backend.tf.nn.relu(-vv)
        ]

    return pde


def func_ones(X):
    x = X[:, 0:1]
    return x * 0 + 1


def func_zeros(X):
    """PDE setting output variable to 0. (used for wall boundary conditions)
    """
    x = X[:, 0:1]
    return x * 0
