clc
clear

sigma = 10 * eye(4);
F = [0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0];
G = [1; 0; 0; 0];
H = [0.8; 1.5; 0; 0];

Q = 1;
S = 1;
R = 1.1;

for i=1:21
   k = ( F*sigma*H+ G*S ) * inv(H'*sigma*H+R);
   sigma = (F-k*H')*sigma*((F-k*H')') + [G -k]*[Q S; S R]*[G'; -k'];
end

fprintf('After 21 iterations sigma is: \n');
disp(sigma);

MSE = [0 1 0.8 1.5]*sigma*[0; 1; 0.8; 1.5];
fprintf('The resulting MSE is: %f \n', MSE);