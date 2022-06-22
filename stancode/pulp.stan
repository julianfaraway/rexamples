data {
  int<lower=0> N;
  int<lower=0> J;
  int<lower=1,upper=J> predictor[N];
  vector[N] response;
}
parameters {
  vector[J] eta;
  real mu;
  real<lower=0> sigmaalpha;
  real<lower=0> sigmaepsilon;
}
transformed parameters {
  vector[J] a;
  vector[N] yhat;

  a = mu + sigmaalpha * eta;

  for (i in 1:N)
    yhat[i] = a[predictor[i]];
}
model {
  eta ~ normal(0, 1);

  response ~ normal(yhat, sigmaepsilon);
}
