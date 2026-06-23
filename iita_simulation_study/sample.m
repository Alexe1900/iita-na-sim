function [pat,freq,samp] = sample(model,sz)

% Simulation of a sample with the BLIM model.
%
% Return the matrix patterns x items without repetitions, a vector with the 
% same numerosity of the patterns, and finally the matrix sx times items
% containing all the generated response pattern with repetition
%
% [PAT,FREQ,SAMP]=SAMPLE(BETA,ETA,PJ,STATES,SZ)
%
% [PAT,FREQ,~]=SAMPLE(BETA,ETA,PJ,STATES,SZ)
%
% where:
% - model is the outcome of an application of the BLIM
% - sz is the sample size
%
% Author: Luca Stefanutti (luca.stefanutti@unipd.it)

if nargin < 2
    sz = sum(model.freq);
end

nitems = size(model.states,2);

if ~isfield(model,'miss')
    model.miss=zeros(sz,nitems);
end

if ~isfield(model,'classes')
    model.classes=eye(nitems);
end

[M,fm] = patfreq(model.miss);
pm = fm/sz;

beta = model.classes'*model.beta;
eta = model.classes'*model.eta;
pc = cumsum(model.pi);
pmc = cumsum(pm);
samp = zeros(sz,nitems);
wa = model.states;
wb = 1-wa;
for i = 1:sz
    j=find(pc>=rand,1);
    m=find(pmc>=rand,1);
    a=(beta<rand(nitems,1))'.*wa(j,:);
    b=(eta>=rand(nitems,1))'.*wb(j,:);
    samp(i,:)=(a|b);
    samp(i,M(m,:)==1)=NaN;
end
[pat,freq]=patfreq(samp);
end


function [pat,freq]=patfreq(data,fr)

% Pattern frequency function for response patterns with missing values
% (missing values encoded as NaN)

i=1;
[m,n]=size(data);

if nargin < 2
    fr=ones(m,1);
end

pat=[];
freq=[];
data1=data;
data1(isnan(data1))=2;
b=3.^(n-1:-1:0);
d=data1*b';
[sd,j]=sort(d+1);
while i<=length(sd)
    f=0;
    v=sd(i);
    p=data(j(i),:);
    while i<=length(sd)&&sd(i)==v
        f=f+fr(j(i));
        i=i+1;
    end
    pat=[pat;p];
    freq=[freq;f];
end

end
