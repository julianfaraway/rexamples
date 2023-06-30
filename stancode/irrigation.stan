data {
  int<lower=0> N;
  array[N] int<lower=1,upper=8> field;
  array[N] int<lower=1,upper=4> irrigation;
  array[N] int<lower=1,upper=2> variety;
  array[N] real y;
}
transformed data { // need to manually create dummy variables
  vector[N] irmeth2;
  vector[N] irmeth3;
  vector[N] irmeth4;
  vector[N] var2;
  for (i in 1:N) {
    irmeth2[i] = irrigation[i] == 2;
    irmeth3[i] = irrigation[i] == 3;
    irmeth4[i] = irrigation[i] == 4;
    var2[i] = variety[i] == 2;
  }
}
parameters {
  vector[8] eta;
  real mu;
  real ir2;
  real ir3;
  real ir4;
  real va2;
  real<lower=0> sigmaf;
  real<lower=0> sigmay;
}
transformed parameters {
  vector[8] fld;
  vector[N] yhat;

  fld = sigmaf * eta;

  for (i in 1:N)
    yhat[i] = mu+ir2*irmeth2[i]+ir3*irmeth3[i]+ir4*irmeth4[i]+va2*var2[i]+fld[field[i]];

}
model {
  eta ~ normal(0, 1);

  y ~ normal(yhat, sigmay);
}
