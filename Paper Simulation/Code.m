clc
clear 
close all

%Given Information:
phi = [0.75 -1.74 -0.3 0 -0.15; 0.09 0.91 -0.0015 0 -0.008; 0 0 0.95 0 0; 0 0 0 0.55 0; 0 0 0 0 0.905];%Transition Matrix
gamma = [0 0 0; 0 0 0; 24.64 0 0; 0 0.835 0; 0 0 1.83];%Input Matrix
H = [1 0 0 0 1; 0 1 0 1 0];%Output Matrix
Q0 = [0.25 0 0; 0 0.5 0; 0 0 0.75];%Initial Value of Input Covariance Matrix
R0 = [0.4 0; 0 0.6];%Initial Value of Noise Covariance Matrix
N = 950;%Number of Samples
M0 = 1000 * eye(5);%Initial Value of Error Covariance Matrix

%Input Vector's Attributes and Simulation:
meanU = [0; 0; 0];
sigmaU = eye(3);
u = mvnrnd(meanU, sigmaU, N);
u = u.';

%Noise Vector's Attributes and Simulation:
meanV = [0; 0];
sigmaV = eye(2);
v = mvnrnd(meanV, sigmaV, N);
v = v.';

%Initial State:
x0 = normrnd(0, 1, [5,1]);

%State and Measurement Simulation:
x = zeros(5,N);%State Vector
x(:,1) = x0;
z = zeros(2,N);%Measurement Vector
for i=1:N-1
    x(:,i+1) = phi * x(:,i) + gamma * u(:,i);
    z(:,i) = H * x(:,i) + v(:,i);
end

%Suboptimal Kalman Filter:
M = M0;
R = R0;
Q = Q0;
xHat = zeros(5,N);%State Estimator
nu = zeros(2,N);%Innovation Sequence

%Kalman Filter's Updating Equations:
for i=1:N-1
   K = M * (H.')/(H*M*(H.')+R);
   M = phi * (M - K*H*M) * (phi.') + gamma * Q * (gamma.');
   nu(:,i) = z(:,i) - H * xHat(:,i);
   xHat(:,i+1) = phi * (xHat(:,i)+ K * z(:,i) - K * H * xHat(:,i));
end

%Calculating the Approximation of Autocorrelation of Innovation Sequence:
CHatDiag = zeros(40,2);
for k = 1:40
   temp = zeros(2);
   for i = k:N
       temp = temp + nu(:,i) * nu(:,i-k+1).';
   end
   CHatDiag(k,1) = (1/N) * temp(1,1);
   CHatDiag(k,2) = (1/N) * temp(2,2);
end

%ro Calculations for Different Values of k:
ro = zeros(40,2);
ro(:,1) = CHatDiag(:,1)/CHatDiag(1,1);%Normalized Autocorrelation Coefficient of the First Diagonal Element
ro(:,2) = CHatDiag(:,2)/CHatDiag(1,2);%Normalized Autocorrelation Coefficient of the Second Diagonal Element

%First Diagonal Element:
figure
myColor1 = [88 59 209]/255;
myColor2 = [172 157 232]/255;
stem(2:40, ro(2:end,1),'.' ,'Color', myColor1);
hold on
plot(2:40, 1.96/sqrt(N)*ones(39,1), '--', 'Color', myColor2);%95 Percent Confidence Upper Limit
plot(2:40, -1.96/sqrt(N)*ones(39,1), '--', 'Color', myColor2);%95 Percent Confidence Lower Limit
xlim([2 40]);
ylim([-0.2 0.2]);
xlabel('k');
ylabel('Normalized Values');
legend('[ro_k]_1_1', 'Confidence Interval Limits');
title("Normalized Autocorrelation Coefficient of the First Diagonal Element of Suboptimal Filter"); 

%Number of Samples Outside 95 Percent Confidence Interval:
counter = 0;
for i=1:40
    if abs(ro(i,1)) > 1.96/sqrt(N)
       counter = counter + 1; 
    end
end
fprintf('The number of samples outside 95 percent confidence interval for the first diagonal element of suboptimal filter is %d \n',counter);

%Second Diagonal Element:
figure
stem(2:40, ro(2:end,2),'.' ,'Color', myColor1);
hold on
plot(2:40, 1.96/sqrt(N)*ones(39,1), '--', 'Color', myColor2);%95 Percent Confidence Upper Limit
plot(2:40, -1.96/sqrt(N)*ones(39,1), '--', 'Color', myColor2);%95 Percent Confidence Lower Limit
xlim([2 40]);
ylim([-0.2 0.2]);
xlabel('k');
ylabel('Normalized Values');
legend('[ro_k]_2_2', 'Confidence Interval Limits');
title("Normalized Autocorrelation Coefficient of the Second Diagonal Element of Suboptimal Filter"); 

%Number of Samples Outside 95 Percent Confidence Interval:
counter = 0;
for i=1:40
    if abs(ro(i,2)) > 1.96/sqrt(N)
       counter = counter + 1; 
    end
end
fprintf('The number of samples outside 95 percent confidence interval for the second diagonal element of suboptimal filter is %d \n',counter);

%%
%Calculating Approximarion of Q and R for the Same Batch of Data:
syms q1Hat q2Hat q3Hat left1 left2 left3 right1 right2 right3

%Initial Values:
M = M0;
R = R0;
Q = Q0;

r1 = zeros(4,1);
r2 = zeros(4,1);
q1 = zeros(4,1);
q2 = zeros(4,1);
q3 = zeros(4,1);
L = zeros(4,1);
estiMSE = zeros(4,1);
MSE = zeros(4,1);

leftEquations = [left1; left2; left3];
rightEquations = [right1; right2; right3];

%Iteration on the Same Data Batch 
for iter=1:4
   
    xHat = zeros(5,N);%State Estimator
    nu = zeros(2,N);%Innovation Sequence
    %Kalman Filter's Updating Equations:
    for i=1:N-1
       K = M * (H.')/(H*M*(H.')+R);
       M = phi * (M - K*H*M) * (phi.') + gamma * Q * (gamma.');
       nu(:,i) = z(:,i) - H * xHat(:,i);
       xHat(:,i+1) = phi * (xHat(:,i)+ K * z(:,i) - K * H * xHat(:,i));
    end

    %Calculating an Estimate of Actual Mean-Square Error:
    temp = 0;
    for i=1:N
       temp = temp + ((x(:,i)-xHat(:,i)).') * (x(:,i)-xHat(:,i));
    end
    estiMSE(iter,1) = (1/N) * temp;

    %Calculating MSE Based on Up-to-Now Updates:
    MSE(iter,1) = trace(M);

    %Likelihood Function Calculation:
    temp = 0;
    for i=1:N
        temp = temp + (nu(:,i).') * inv(H*M*(H.')+R) * nu(:,i);
    end
    L(iter,1) = (-1/N) * temp  - log(det(H*M*(H.')+R));

    %Calculating the Approximation of Autocorrelation of Innovation Sequence:
    CHat = zeros(12,2);
    for n=1:2:11
        temp = zeros(2);
        for i=floor(n/2):N-1
            temp = temp + nu(:,i+1) * nu(:,i-floor(n/2)+1).';
        end
        CHat(n:n+1,:) = (1/N) * temp;
    end

    %Caclculating an Estimate of MHT:
    A = [H*phi; H*phi*(eye(5)-K*H)*phi; H*(phi*(eye(5)-K*H))^2*phi; H*(phi*(eye(5)-K*H))^3*phi; H*(phi*(eye(5)-K*H))^4*phi];
    APsuedo = pinv(A);
    MHTHat = K * CHat(1:2,:) + APsuedo * CHat(3:end,:);

    for k=1:5
        %Calculating an Estimate of Q:
        QHat = [q1Hat 0 0; 0 q2Hat 0; 0 0 q3Hat];
        %Equation 28 Left-Hand Side:
        left = 0;
        for j=0:k-1
           left = left + H * (phi^j) * gamma * QHat * (gamma.') * ((inv(phi^(k-j))).') * (H.');
        end

        %Equation 28 Right-Hand Side:
        right = 0;
        omegaHat = phi * (-K*(MHTHat.')-MHTHat*(K.')+K*CHat(1:2,:)*(K.')) * (phi.');
        for j =0:k-1
           right = right - H * (phi^j) * omegaHat * ((inv(phi^(k-j))).') * (H.');
        end
        right = right + (MHTHat.') * ((inv(phi^k)).') * (H.') - H * (phi^k) * MHTHat;

        if k==1
            leftEquations(1,1) = left(1,1);
            leftEquations(2,1) = left(2,2);

            rightEquations(1,1) = right(1,1);
            rightEquations(2,1) = right(2,2);

        elseif k==5
            leftEquations(3,1) = left(1,1);

            rightEquations(3,1) = right(1,1);
        end
    end

    %Solving the Linear Equations:
    eq1 = leftEquations(1,1) == rightEquations(1,1);
    eq2 = leftEquations(2,1) == rightEquations(2,1);
    eq3 = leftEquations(3,1) == rightEquations(3,1);

    %Values of q1, q2 and q3:
    solutions = solve([eq1, eq2, eq3], [q1Hat, q2Hat, q3Hat]);
    
    QHatSol = [double(solutions.q1Hat) 0 0; 0 double(solutions.q2Hat) 0; 0 0 double(solutions.q3Hat)];
    q1(iter,1) = double(solutions.q1Hat);
    q2(iter,1) = double(solutions.q2Hat);
    q3(iter,1) = double(solutions.q3Hat);
    
    %Calculating an Estimation of R:
    RHat = CHat(1:2,:) - H * MHTHat;
    RHat = [RHat(1,1) 0; 0 RHat(2,2)];
    r1(iter,1) = RHat(1,1);
    r2(iter,1) = RHat(2,2);
    
    R = RHat;
    Q = QHatSol;
end    

fprintf('The calculated value for r1 for 4 iterations on the same batch of data is: \n');
disp(r1);
fprintf('\n');

fprintf('The calculated value for r2 for 4 iterations on the same batch of data is: \n');
disp(r2);
fprintf('\n');

fprintf('The calculated value for q1 for 4 iterations on the same batch of data is: \n');
disp(q1);
fprintf('\n');

fprintf('The calculated value for q2 for 4 iterations on the same batch of data is: \n');
disp(q2);
fprintf('\n');

fprintf('The calculated value for q3 for 4 iterations on the same batch of data is: \n');
disp(q3);
fprintf('\n');

fprintf('The calculated value for likelihood function for 4 iterations on the same batch of data is: \n');
disp(L);
fprintf('\n');

fprintf('The calculated value for the actual value of MSE for 4 iterations on the same batch of data is: \n');
disp(estiMSE);
fprintf('\n');

fprintf('The calculated value for the calculated MSE for 4 iterations on the same batch of data is: \n');
disp(MSE);
fprintf('\n');

%Check Use Simulation:
QGoal = eye(3);
RGoal = eye(2);

xHat = zeros(5,N);%State Estimator
nu = zeros(2,N);%Innovation Sequence
%Kalman Filter's Updating Equations:
for i=1:N-1
   K = M * (H.')/(H*M*(H.')+RGoal);
   M = phi * (M - K*H*M) * (phi.') + gamma * QGoal * (gamma.');
   nu(:,i) = z(:,i) - H * xHat(:,i);
   xHat(:,i+1) = phi * (xHat(:,i)+ K * z(:,i) - K * H * xHat(:,i));
end

%Calculating an Estimate of Actual Mean-Square Error Based on R and Q of Goal:
temp = 0;
for i=1:N
   temp = temp + ((x(:,i)-xHat(:,i)).') * (x(:,i)-xHat(:,i));
end
estiMSEGoal = (1/N) * temp;

%Calculating MSE Based on Up-to-Now Updates Based on R and Q of Goal:
MSEGoal = trace(M);

%Likelihood Function Calculation Based on R and Q of Goal:
temp = 0;
for i=1:N
    temp = temp + (nu(:,i).') * inv(H*M*(H.')+R) * nu(:,i);
end
LGoal = (-1/N) * temp  - log(det(H*M*(H.')+R));

fprintf('The check set value for likelihood function is: \n');
disp(LGoal);
fprintf('\n');

fprintf('The check set value for the actual value of MSE is: \n');
disp(estiMSEGoal);
fprintf('\n');

fprintf('The check set value for the calculated MSE is: \n');
disp(MSEGoal);
fprintf('\n');

%%

clc
close all
M = M0;
%Kalman Filter's Updating Equations:
for i=1:N-1
   K = M * (H.')/(H*M*(H.')+R);
   M = phi * (M - K*H*M) * (phi.') + gamma * Q * (gamma.');
   nu(:,i) = z(:,i) - H * xHat(:,i);
   xHat(:,i+1) = phi * (xHat(:,i)+ K * z(:,i) - K * H * xHat(:,i));
end

%Calculating the Approximation of Autocorrelation of Innovation Sequence:
CHatDiag = zeros(40,2);
for k = 1:40
   temp = zeros(2);
   for i = k:N
       temp = temp + nu(:,i) * nu(:,i-k+1).';
   end
   CHatDiag(k,1) = (1/N) * temp(1,1);
   CHatDiag(k,2) = (1/N) * temp(2,2);
end

%ro Calculations for Different Values of k:
ro = zeros(40,2);
ro(:,1) = CHatDiag(:,1)/CHatDiag(1,1);%Normalized Autocorrelation Coefficient of the First Diagonal Element
ro(:,2) = CHatDiag(:,2)/CHatDiag(1,2);%Normalized Autocorrelation Coefficient of the Second Diagonal Element

%First Diagonal Element:
figure
myColor1 = [88 59 209]/255;
myColor2 = [172 157 232]/255;
stem(2:40, ro(2:end,1),'.' ,'Color', myColor1);
hold on
plot(2:40, 1.96/sqrt(N)*ones(39,1), '--', 'Color', myColor2);%95 Percent Confidence Upper Limit
plot(2:40, -1.96/sqrt(N)*ones(39,1), '--', 'Color', myColor2);%95 Percent Confidence Lower Limit
xlim([2 40]);
ylim([-0.2 0.2]);
xlabel('k');
ylabel('Normalized Values');
legend('[ro_k]_1_1', 'Confidence Interval Limits');
title("Normalized Autocorrelation Coefficient of the First Diagonal Element of Optimal Filter"); 

%Number of Samples Outside 95 Percent Confidence Interval:
counter = 0;
for i=2:40
    if abs(ro(i,1)) > 1.96/sqrt(N)
       counter = counter + 1; 
    end
end
fprintf('The number of samples outside 95 percent confidence interval for the first diagonal element of optimal filter is %d \n',counter);

%Second Diagonal Element:
figure
stem(2:40, ro(2:end,2),'.' ,'Color', myColor1);
hold on
plot(2:40, 1.96/sqrt(N)*ones(39,1), '--', 'Color', myColor2);%95 Percent Confidence Upper Limit
plot(2:40, -1.96/sqrt(N)*ones(39,1), '--', 'Color', myColor2);%95 Percent Confidence Lower Limit
xlim([2 40]);
ylim([-0.2 0.2]);
xlabel('k');
ylabel('Normalized Values');
legend('[ro_k]_2_2', 'Confidence Interval Limits');
title("Normalized Autocorrelation Coefficient of the Second Diagonal Element of Optimal Filter"); 

%Number of Samples Outside 95 Percent Confidence Interval:
counter = 0;
for i=2:40
    if abs(ro(i,2)) > 1.96/sqrt(N)
       counter = counter + 1; 
    end
end
fprintf('The number of samples outside 95 percent confidence interval for the second diagonal element of optimal filter is %d \n',counter);

%%
clc
clear 
close all

%Given Information:
phi = [0.75 -1.74 -0.3 0 -0.15; 0.09 0.91 -0.0015 0 -0.008; 0 0 0.95 0 0; 0 0 0 0.55 0; 0 0 0 0 0.905];%Transition Matrix
gamma = [0 0 0; 0 0 0; 24.64 0 0; 0 0.835 0; 0 0 1.83];%Input Matrix
H = [1 0 0 0 1; 0 1 0 1 0];%Output Matrix
Q0 = [0.25 0 0; 0 0.5 0; 0 0 0.75];%Initial Value of Input Covariance Matrix
R0 = [0.4 0; 0 0.6];%Initial Value of Noise Covariance Matrix
N = 950;%Number of Samples
M0 = 1000 * eye(5);%Initial Value of Error Covariance Matrix

%Input Vector's Attributes and Simulation:
meanU = [0; 0; 0];
sigmaU = eye(3);
u = mvnrnd(meanU, sigmaU, 10*N);
u = u.';

%Noise Vector's Attributes and Simulation:
meanV = [0; 0];
sigmaV = eye(2);
v = mvnrnd(meanV, sigmaV, 10*N);
v = v.';

%Initial State:
x0 = normrnd(0, 1, [5,1]);

%State and Measurement Simulation:
x = zeros(5,10*N);%State Vector
x(:,1) = x0;
z = zeros(2,10*N);%Measurement Vector
for i=1:10*N-1
    x(:,i+1) = phi * x(:,i) + gamma * u(:,i);
    z(:,i) = H * x(:,i) + v(:,i);
end

%Suboptimal Kalman Filter:
M = M0;
R = R0;
Q = Q0;

r1 = zeros(10,1);
r2 = zeros(10,1);
q1 = zeros(10,1);
q2 = zeros(10,1);
q3 = zeros(10,1);
L = zeros(10,1);
estiMSE = zeros(10,1);
MSE = zeros(10,1);

%Iteration on 10 Data Batches
for iter=1:10
    
    xHat = zeros(5,N);%State Estimator
    nu = zeros(2,N);%Innovation Sequence

    %Kalman Filter's Updating Equations:
    for i=(iter-1)*N+1:iter*N-1
       K = M * (H.')/(H*M*(H.')+R);
       M = phi * (M - K*H*M) * (phi.') + gamma * Q * (gamma.');
       nu(:,i-(iter-1)*N) = z(:,i) - H * xHat(:,i-(iter-1)*N);
       xHat(:,i+1-(iter-1)*N) = phi * (xHat(:,i-(iter-1)*N)+ K * z(:,i) - K * H * xHat(:,i-(iter-1)*N));
    end

    %Calculating Approximarion of Q and R for the Same Batch of Data:
    syms q1Hat q2Hat q3Hat left1 left2 left3 right1 right2 right3

    leftEquations = [left1; left2; left3];
    rightEquations = [right1; right2; right3];

    %Calculating an Estimate of Actual Mean-Square Error:
    temp = 0;
    for i=1:N
       temp = temp + ((x(:,i)-xHat(:,i)).') * (x(:,i)-xHat(:,i));
    end
    estiMSE(iter,1) = (1/N) * temp;

    %Calculating MSE Based on Up-to-Now Updates:
    MSE(iter,1) = trace(M);

    %Likelihood Function Calculation:
    temp = 0;
    for i=1:N
        temp = temp + (nu(:,i).') * inv(H*M*(H.')+R) * nu(:,i);
    end
    L(iter,1) = (-1/N) * temp  - log(det(H*M*(H.')+R));

    %Calculating the Approximation of Autocorrelation of Innovation Sequence:
    CHat = zeros(12,2);
    for n=1:2:11
        temp = zeros(2);
        for i=floor(n/2):N-1
            temp = temp + nu(:,i+1) * nu(:,i-floor(n/2)+1).';
        end
        CHat(n:n+1,:) = (1/N) * temp;
    end

    %Caclculating an Estimate of MHT:
    A = [H*phi; H*phi*(eye(5)-K*H)*phi; H*(phi*(eye(5)-K*H))^2*phi; H*(phi*(eye(5)-K*H))^3*phi; H*(phi*(eye(5)-K*H))^4*phi];
    APsuedo = pinv(A);
    MHTHat = K * CHat(1:2,:) + APsuedo * CHat(3:end,:);

    for k=1:5
        %Calculating an Estimate of Q:
        QHat = [q1Hat 0 0; 0 q2Hat 0; 0 0 q3Hat];
        %Equation 28 Left-Hand Side:
        left = 0;
        for j=0:k-1
           left = left + H * (phi^j) * gamma * QHat * (gamma.') * ((inv(phi^(k-j))).') * (H.');
        end

        %Equation 28 Right-Hand Side:
        right = 0;
        omegaHat = phi * (-K*(MHTHat.')-MHTHat*(K.')+K*CHat(1:2,:)*(K.')) * (phi.');
        for j =0:k-1
           right = right - H * (phi^j) * omegaHat * ((inv(phi^(k-j))).') * (H.');
        end
        right = right + (MHTHat.') * ((inv(phi^k)).') * (H.') - H * (phi^k) * MHTHat;

        if k==1
            leftEquations(1,1) = left(1,1);
            leftEquations(2,1) = left(2,2);

            rightEquations(1,1) = right(1,1);
            rightEquations(2,1) = right(2,2);

        elseif k==5
            leftEquations(3,1) = left(1,1);

            rightEquations(3,1) = right(1,1);
        end
    end

    %Solving the Linear Equations:
    eq1 = leftEquations(1,1) == rightEquations(1,1);
    eq2 = leftEquations(2,1) == rightEquations(2,1);
    eq3 = leftEquations(3,1) == rightEquations(3,1);
    
    %VAlues of q1, q2, and q3:
    solutions = solve([eq1, eq2, eq3], [q1Hat, q2Hat, q3Hat]);
    
    %Calculating an Estimate of R:
    RHat = CHat(1:2,:) - H * MHTHat;
    
    %Updating R an Q's Estimators Based on Last Iteration Values:
    if iter==1
        q1(iter,1) = Q0(1,1) + (double(solutions.q1Hat)-Q0(1,1))/(iter+1);
        q2(iter,1) = Q0(2,2) + (double(solutions.q2Hat)-Q0(2,2))/(iter+1);
        q3(iter,1) = Q0(3,3) + (double(solutions.q3Hat)-Q0(3,3))/(iter+1);
        
        r1(iter,1) = R0(1,1) + (RHat(1,1)-R0(1,1))/(iter+1);
        r2(iter,1) = R0(2,2) + (RHat(2,2)-R0(2,2))/(iter+1);
        
    else
        q1(iter,1) = q1(iter-1,1) + (double(solutions.q1Hat)-q1(iter-1,1))/(iter+1);
        q2(iter,1) = q2(iter-1,1) + (double(solutions.q2Hat)-q2(iter-1,1))/(iter+1);
        q3(iter,1) = q3(iter-1,1) + (double(solutions.q3Hat)-q3(iter-1,1))/(iter+1);
        
        r1(iter,1) = r1(iter-1,1) + (RHat(1,1)-r1(iter-1,1))/(iter+1);
        r2(iter,1) = r2(iter-1,1) + (RHat(2,2)-r2(iter-1,1))/(iter+1); 
    end

    R = [r1(iter,1) 0; 0 r2(iter,1)];
    Q = [q1(iter,1) 0 0; 0 q2(iter,1) 0; 0 0 q3(iter,1)];

end


myColor1 = [88 59 209]/255;
myColor2 = [172 157 232]/255;

batchIndex = 0:10;
batchLen = length(batchIndex);

figure
r1 = [0.4; r1];
plot(batchIndex, r1,'.-' ,'Color', myColor1);
hold on
plot(batchIndex, ones(batchLen,1), '--', 'Color', myColor2);
hold off
xlabel('Batch Number');
legend('Calculated Value of r_1', 'Actual Value of r_1');

figure
r2 = [0.6; r2];
plot(batchIndex, r2,'.-' ,'Color', myColor1);
hold on
plot(batchIndex, ones(batchLen,1), '--', 'Color', myColor2);
hold off
xlabel('Batch Number');
legend('Calculated Value of r_2', 'Actual Value of r_2');

figure
q1 = [0.25; q1];
plot(batchIndex, q1,'.-' ,'Color', myColor1);
hold on
plot(batchIndex, ones(batchLen,1), '--', 'Color', myColor2);
hold off
xlabel('Batch Number');
legend('Calculated Value of q_1', 'Actual Value of q_1');

figure
q2 = [0.5; q2];
plot(batchIndex, q2,'.-' ,'Color', myColor1);
hold on
plot(batchIndex, ones(batchLen,1), '--', 'Color', myColor2);
hold off
xlabel('Batch Number');
legend('Calculated Value of q_2', 'Actual Value of q_2');

figure
q3 = [0.75; q3];
plot(batchIndex, q3,'.-' ,'Color', myColor1);
hold on
plot(batchIndex, ones(batchLen,1), '--', 'Color', myColor2);
hold off
xlabel('Batch Number');
legend('Calculated Value of q_3', 'Actual Value of q_3');

