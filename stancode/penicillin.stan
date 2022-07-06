data {
  int<lower=0> N;
  int<lower=0> Nt;
  int<lower=0> Nb;
  int<lower=1,upper=Nt> treat[N];
  int<lower=1,upper=Nb> blk[N];
  vector[N] y;
}
parameters {
  vector[Nb] eta;
  vector[Nt] trt;
  real<lower=0> sigmablk;
  real<lower=0> sigmaepsilon;
}
transformed parameters {
  vector[Nb] bld;
  vector[N] yhat;

  bld <- sigmablk * eta;

  for (i in 1:N)
    yhat[i] <- trt[treat[i]]+bld[blk[i]];

}
model {
  eta ~ normal(0, 1);

  y ~ normal(yhat, sigmaepsilon);
}
