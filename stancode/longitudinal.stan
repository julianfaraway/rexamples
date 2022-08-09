data {
     int<lower=0> Nobs;
     int<lower=0> Npreds;
     int<lower=0> Ngroups;
     vector[Nobs] y;
     matrix[Nobs,Npreds] x;
     vector[Nobs] timevar;
     int<lower=1,upper=Ngroups> group[Nobs];
}
parameters {
           vector[Npreds] beta;
           real<lower=0> sigmaint;
           real<lower=0> sigmaslope;
           real<lower=0> sigmaeps;

           vector[Ngroups] etaint;
           vector[Ngroups] etaslope;
}
transformed parameters {
  vector[Ngroups] ranint;
  vector[Ngroups] ranslope;
  vector[Nobs] yhat;

  ranint  = sigmaint * etaint;
  ranslope  = sigmaslope * etaslope;

  for (i in 1:Nobs)
    yhat[i] = x[i]*beta+ranint[group[i]]+ranslope[group[i]]*timevar[i];

}
model {
  etaint ~ normal(0, 1);
  etaslope ~ normal(0, 1);
  y ~ normal(yhat, sigmaeps);
}
