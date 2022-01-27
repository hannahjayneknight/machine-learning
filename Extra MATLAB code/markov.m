% Markov chain 

% Two states A, B 
% Probability of A to B 0.5
% Probability of A to A 0.5
% Probability of B to A 0.1
% Probability of B to B 0.9

pA = [0.5 0.5];
pB = [0.1 0.9];

S_A = 0;
S_B = 0;
N = 10000000;
STATE_A = 1;
STATE_B = 0;

count = 0;
% lets do N trials starting from A

for i = 1:N
    %if M > 0;
    count = count+1;
    %end;
    p = rand(1);
    
    if STATE_A == 1;
      if p < pA(1)
          STATE_B = 0;
          STATE_A = 1;
          %if count == M;
          S_A = S_A+1;
          %end;
      else
          STATE_B = 1;
          STATE_A = 0;
          %if count == M;
          S_B = S_B+1;
          %end;
      end;    
          
    elseif STATE_B == 1;
        if p < pB(2)
          STATE_B = 1;
          STATE_A = 0;
          %if count == M;
          S_B = S_B+1;
          %end;
      else
          STATE_B = 0;
          STATE_A = 1;
          %if count == M;
          S_A = S_A+1;
          %end;
      end;   
    end;
     
      
        
    
end
    
    % Probabilities of being in each state
    
    [S_A,S_B]/(S_A+S_B)
    
   