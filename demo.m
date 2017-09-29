% example of HiDDen algorithm on unipartite graph
load('coauthor_net.mat');
X = hidden(A); K = 10;
sub_size=zeros(K,1); subgraph=cell(K,1); density=zeros(K,1);
for i=1:K
idx=find(X(:,i)>0.5);
subgraph{i}=A(idx,idx);
n1=length(idx);
density(i)=nnz(subgraph{i})/(n1^2-n1);
sub_size(i)=n1;
end

% example of HiDDen algorithm on bipartite graph
load('Infutor_result_alternative.mat');
[X, Y] = hidden_bipartite(A);
sub_size=zeros(K,2); subgraph=cell(K,1); density=zeros(K,1);

for i=1:K
    idx=find(X(:,i)>0.1); idy=find(Y(:,i)>0.1);
    n1=length(idx); n2=length(idy);
    sub_size(i,1)=n1; sub_size(i,2)=n2;
    S=A(idx,idy);
    subgraph{i}=S;
    density(i)=nnz(S)/(n1*n2);
end

% example of HiDDen on-query algorithm
load('coauthor_net.mat');
K = 10;
X = hidden_query(A, 32774);
sub_size=zeros(K,1); subgraph=cell(K,1); density=zeros(K,1);
for i=1:K
idx=find(X(:,i)>0.5);
subgraph{i}=A(idx,idx);
n1=length(idx);
density(i)=nnz(subgraph{i})/(n1^2-n1);
sub_size(i)=n1;
end