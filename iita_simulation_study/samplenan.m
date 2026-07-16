function sample=samplenan(model,sz)

% SAMPLE simulation of a sample with the missBLIM model

n=length(model.beta);
pc=cumsum(model.pi);
sample.data=zeros(sz,n);
sample.states=zeros(sz,n);

w1=model.states;
w0=1-model.states;

for i=1:sz
    j=min(find(pc>=rand));
    sample.states(i,:)=model.states(j,:);
    
    r0=rand(n,1);
    x0m=(r0<=model.mu0)'.*w0(j,:);
    x01=(model.mu0<r0 & r0 <= model.mu0+model.eta)'.*w0(j,:);
    r1=rand(n,1);
    x1m=(r1<=model.mu1)'.*w1(j,:);
    x11=(r1>model.mu1+model.beta)'.*w1(j,:);
    sample.data(i,x0m|x1m)=nan;
    sample.data(i,x01|x11)=1;
end

[sample.pat,sample.freq]=patfreq(sample.data);

