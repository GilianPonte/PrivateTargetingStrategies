data_simulation = function(n) {
  x = stats::model.matrix(~.-1, data.frame("covariate_1" = rnorm(n), "covariate_2"= rnorm(n), "covariate_3" = rnorm(n), "covariate_4" = rnorm(n), "covariate_5" = rnorm(n), "covariate_6" = rnorm(n)))
  p = 0.5
  w = as.numeric(rbinom(n,1,p)==1)
  m = pmax(0, x[,1] + x[,2], x[,3]) + pmax(0, x[,4] + x[,5])
  tau = x[,1] + log(1 + exp(x[,2]))^2
  mu1 = m + tau/2
  mu0 = m - tau/2
  y = w*mu1 + (1-w) * mu0 + 0.5*rnorm(n)
  list(x=x, w=w, y=y, p=p, m=m, mu0=mu0, mu1=mu1, tau=tau)
}

