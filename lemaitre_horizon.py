import sympy as sp;
import sympy.diffgeom as spdg
import math 

# Declare Lemaitre coordinate system on a R^4 manifold
tau = sp.symbols('tau', real=True)
rho, theta, phi = sp.symbols('rho theta phi', real=True)
r = (3/2*(rho-tau))**(2/3)

m = spdg.Manifold('M', 4)
p = spdg.Patch('P', m)
Lemaitre = spdg.CoordSystem('Lemaitre', p, (tau, rho, theta, phi))

dtau, drho, dtheta, dphi = Lemaitre.base_oneforms()

# Declare Lemaitre metric 
metric = -spdg.TensorProduct(dtau, dtau) + 1/r*spdg.TensorProduct(drho, drho) + r**2*spdg.TensorProduct(dtheta, dtheta) + r**2*sp.sin(theta)**2*spdg.TensorProduct(dphi, dphi)

# Compute Riemann tensor R^rho_sigma_mu_nu associated to Lemaitre metric 
riemann = spdg.metric_to_Riemann_components(metric, (tau, rho, theta, phi))

# Evaluta metric and Riemann tensor on a particular event on the horizon where the metric is Minkowski metric
event = {tau: 1/3, rho: 1, theta: math.pi/2, phi: 0}
metric_eval = spdg.twoform_to_matrix(metric).subs(event)
riemann_eval = riemann.subs(event)
print(spdg.twoform_to_matrix(metric).subs(event))

# Compute fully-contravariant Riemann tensor R^rho^sigma^mu^nu
riemann_eval2 = riemann_eval.as_mutable()
for i in range(0,4):
    for j in range(0,4):
        for k in range(0,4):
            for l in range(0,4):
                riemann_eval2[i,j,k,l] = 0
                for jprime in range(0,4):
                    for kprime in range(0,4):
                        for lprime in range(0,4):
                            riemann_eval2[i,j,k,l] += metric_eval[j,jprime]*metric_eval[k,kprime]*metric_eval[l,lprime]*riemann_eval[i,jprime,kprime,lprime]
riemann_eval = riemann_eval2
print(riemann_eval)

# Compute Kretschmann scalar R_rho_sigma_mu_nu*R^rho^sigma^mu^nu, which is known to be 12 on the horizon
scalar = 0
for i in range(0,4):
    for j in range(0,4):
        for k in range(0,4):
            for l in range(0,4):
                for iprime in range(0,4):
                    for jprime in range(0,4):
                        for kprime in range(0,4):
                            for lprime in range(0,4):
                                scalar += metric_eval[i,iprime]*metric_eval[j,jprime]*metric_eval[k,kprime]*metric_eval[l,lprime]*riemann_eval[iprime,jprime,kprime,lprime]*riemann_eval[i,j,k,l];
print(scalar)
