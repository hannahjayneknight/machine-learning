eps = 0.00001;


A = [
    
0.5 0.5 0 0 0 eps;
0.5-eps 0.5 0 0 0 0;
0 0 0.7 0.3-eps eps 0;
0 0 0.5 0.5 0 0;
0 0 0 eps 0.2-eps 0.8;
eps 0 0 0 0.3-eps 0.7

]

[e,v] = eig(A)

