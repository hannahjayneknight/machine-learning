
clear all
close all

x = [0:0.5:100];
l = length(x);
y = [];
ys = [];
k = 10;

for i = 1:l
    y = [y,2.7*x(i)+100+k*(rand(1)-rand(1))];
end;


plot(x,y,'*')
hold on
xlabel('x')
ylabel('y')
title('Validation set');
grid on



 U = [x',1+0*x'];
 Y = y';
 M = U'*U
 Z = U'*Y
 theta = inv(U'*U)*(U'*Y)
 
 
%%%%%%%%%%%%% Gradient descent %%%%%%%

epochs = 100000;
m = 2.7;
c = 90;
alpha = 1e-4;
gradient_m = [];
gradient_c = [];
cost_epoch = [];
m_val = [];
c_val = [];

for i = 1:epochs
    % calculate partial derivatives
    grad_m = 0;
    grad_c = 0;
    cost = 0; 
    for j = 1:l
        grad_m = grad_m + 2*(m*x(j)+c-y(j))*x(j);
        grad_c = grad_c + 2*(m*x(j)+c-y(j));
        cost = cost + (y(j) - m*x(j)-c)^2; 
    end;
    grad_m = grad_m/l;
    grad_c = grad_c/l;
    cost = cost/l;
    gradient_m = [gradient_m, grad_m];
    gradient_c = [gradient_c, grad_c];
    cost_epoch = [cost_epoch,cost];
    c = c-alpha*grad_c;
    m = m-alpha*grad_m;
    m_val = [m_val,m];
    c_val = [c_val,c];
    %pause
end; 
figure
plot(cost_epoch)
grid on
figure
plot(m_val)
xlabel('epoch');
ylabel('m');
title('Estimate of the parameter m');
grid on
figure
plot(c_val)
xlabel('epoch');
ylabel('c');
title('Estimate of the parameter c');
grid on
figure
plot(cost_epoch)
xlabel('epoch');
ylabel('cost');
title('Evolution of cost');
grid on
%axis([0 10000 0 100000])

    
        



 



