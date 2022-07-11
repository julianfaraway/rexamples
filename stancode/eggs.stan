data {
     int<lower=0> Nobs;
     int<lower=0> Nlev1;
     int<lower=0> Nlev2;
     int<lower=0> Nlev3;
     vector[Nobs] y;
     int<lower=1,upper=Nlev1> levind1[Nobs];
     int<lower=1,upper=Nlev2> levind2[Nobs];
     int<lower=1,upper=Nlev3> levind3[Nobs];
     real<lower=0> sdscal;
}
parameters {
           real mu;
           real<lower=0> sigmalev1;
           real<lower=0> sigmalev2;
           real<lower=0> sigmalev3;
           real<lower=0> sigmaeps;

           vector[Nlev1] eta1;
           vector[Nlev2] eta2;
           vector[Nlev3] eta3;
}
transformed parameters {
  vector[Nlev1] ran1;
  vector[Nlev2] ran2;
  vector[Nlev3] ran3;
  vector[Nobs] yhat;

  ran1  = sigmalev1 * eta1;
  ran2  = sigmalev2 * eta2;
  ran3  = sigmalev3 * eta3;

  for (i in 1:Nobs)
    yhat[i] = mu+ran1[levind1[i]]+ran2[levind2[i]]+ran3[levind3[i]];

}
model {
  eta1 ~ normal(0, 1);
  eta2 ~ normal(0, 1);
  eta3 ~ normal(0, 1);
  sigmalev1 ~ cauchy(0, 2.5*sdscal);
  sigmalev2 ~ cauchy(0, 2.5*sdscal);
  sigmalev3 ~ cauchy(0, 2.5*sdscal);
  sigmaeps ~ cauchy(0, 2.5*sdscal);
  y ~ normal(yhat, sigmaeps);
}
