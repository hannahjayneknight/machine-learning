
clear all
close all

x = [0:1:100];
l = length(x);

y = [];
reg = [];
k = 50; % amplitude of noise

centres = [0:10:100];
width = 20; %20


for i = 1:l
    y = [y,2.7*x(i)+100+k*(rand(1)-rand(1))];
    reg = [reg; [exp(-(x(i)-centres(1:10)).^2/width.^2),1]];   
    %ys = [ys, 2.7*x(i)+100];  
end;


m = exp(-(x-centres(5)).^2/width.^2);





% linear parameters

 U = [x',1+0*x'];
 Y = y';
 M = U'*U
 Z = U'*Y
 theta = inv(U'*U)*(U'*Y)
 
 % RBF parameters
 
 theta_r = inv(reg'*reg)*reg'*Y;
 y_m = reg*theta_r;
 y_l = U*theta;
 
figure 
plot(x,y,'+')
grid on


figure 
plot(x,y,'+')
xlabel('x')
ylabel('y')
grid on


%figure
hold on
plot(x,50*reg(:,1:10))
grid on
xlabel('x')
%ylabel('y')
title('50xBasis functions');



figure


plot(x,y,'+')
xlabel('x')
ylabel('y')
hold on
plot(x,y_m)
%plot(x,y_l,'b')
hold on
xlabel('x')
ylabel('y')
title('Radial Basis Function Approximation');
grid on

figure
hold on
plot(x,reg)
grid on
xlabel('x')
ylabel('y')
title('Basis functions');

figure
%plot(x,y,'+')
hold on
plot(x,y_m,'r')
plot(x,y_l,'b')
hold on
xlabel('x')
ylabel('y')
title('Linear and RBF Approx');
grid on



 
 
 
 


 
