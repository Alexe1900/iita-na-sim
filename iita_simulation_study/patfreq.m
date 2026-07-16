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