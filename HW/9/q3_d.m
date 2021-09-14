clc
clear

sigma = 10 * eye(3);
F = [0.2 0 0.1; 1 0 0; 0 0 0];
G = [1; 0; 1];
H = [1; 0; 0];

Q = 1;
S = 0;
R = 0;

for i=1:4
   k = ( F*sigma*H+ G*S ) * inv(H'*sigma*H+R);
   sigma = (F-k*H')*sigma*((F-k*H')') + [G -k]*[Q S; S R]*[G'; -k'];
end

fprintf('After 4 iterations sigma is: \n');
disp(sigma);

a = [1 0 0];
MSE = a*sigma*a';
fprintf('The resulting MSE is: %f \n', MSE);