/*
Latin square style design
*/
data {
  int<lower=0> N;
  int<lower=0> Nt;
  array[N] int<lower=1,upper=Nt> treat;
  array[N] int<lower=1,upper=Nt> blk1;
  array[N] int<lower=1,upper=Nt> blk2;
  array[N] real y;
  real<lower=0> sdscal;
}
parameters {
  vector[Nt] eta1;
  vector[Nt] eta2;
  vector[Nt] trt;
  real<lower=0> sigmab1;
  real<lower=0> sigmab2;
  real<lower=0> sigmaeps;
}
transformed parameters {
  vector[Nt] bld1;
  vector[Nt] bld2;
  vector[N] yhat;

  bld1 = sigmab1 * eta1;
  bld2 = sigmab2 * eta2;

  for (i in 1:N)
    yhat[i] = trt[treat[i]] + bld1[blk1[i]] + bld2[blk2[i]];

}
model {
  eta1 ~ normal(0, 1);
  eta2 ~ normal(0, 1);
  sigmab1 ~ cauchy(0, 2.5*sdscal);
  sigmab2 ~ cauchy(0, 2.5*sdscal);
  sigmaeps ~ cauchy(0, 2.5*sdscal);

  y ~ normal(yhat, sigmaeps);
}
